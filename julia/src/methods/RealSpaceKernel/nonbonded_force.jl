"""
1. Spatially sort atoms (done every M steps)
	- Divide space into square voxels of width w, (similar or smaller than r_cut)
        - Calculate centroid of molecule and assign to voxel (for particles this is trival)
        - Trace through voxels in a spatially contiguous order and generate new ordering for     		atoms
2. Tile atoms into blocks of 32
	- Build bounding boxes around the 32 atoms in each tile O(N)
3. Figure out which tiles interact with eachother, O(N^2) if done with loops
	-Nearest distance between boxes is < r_cut + skin distance
        - Done every Q steps, if Q != 1 you need a skin distance
4. For each pair of interacting blocks compute distance of bounding box to each atom in the second block. If distance > r_cut set flag for that indicating no interacctions need to be calculated
5. If less than 12 atoms of 32 in a tile have interactions use the flags:
	- For each of the 32 threads in a tile, each thread computes the interactions between 		one atom in the first block with all atoms in the second block
	- This requires a reduction to calculate the force
6. If more than 12 of the 32 in a tile have interactions do not use the flags:
	-Compute all interactions, presumably there is way to do this without reduction

- For coulomb force r_cut_real to be <L/2 so we can use normal force loop
"""

#&how u do this without warp divergence??
function nearest_mirror!(r_ij, box_sizes)
    #r_ij > L/2 --> sub L
    #r_ij < -L/2 -- add L
    r_ij += sign()
end

function full_tile_kernel(r, i_offset, j_offset, box_sizes, tid::Int32, 
    tile_forces_i::CuArray{Float32, 3}, tile_forces_j::CuArray{Float32, 3}, force::Function)
     #Start loop in each thread at a different place
    #* does this cause warp divergence?
    for j in tid:(tid + WARP_SIZE) #*is THREADIDX.X 1 iNdexed???
        wrapped_j_idx = (j - 1) & (ATOM_BLOCK_SIZE - 1) #equivalent to modulo when ATOM_BLOCK_SIZE is power of 2

        r_ij = r[i_offset + tid] .- r[j_offset + wrapped_j_idx]
        nearest_mirror!(r_ij, box_sizes)
        F_ij = force(r_ij)

        #No race conditions as threads in warp execute in step
        tile_forces_i[tile_idx, tid, :] += F_ij
        tile_forces_j[tile_idx, wrapped_j_idx, :] -= F_ij
    end
end

function partial_tile_kernel(r, box_sizes, tid::Int32, tile_forces_i::CuArray{Float32, 3},
         tile_forces_j::CuArray{Float32, 3}, force::Function)
    F_i = 0.0f32
    #Store forces by pairs and reduce after block executes
    F_j = CuStaticSharedArray(Float32, (ATOM_BLOCK_SIZE, ATOM_BLOCK_SIZE, 3))

    
    for j in tile.j_index_range
        if tnl.atom_flags[tile.i, j]
            r_ij = r[i] .- r[j]
            nearest_mirror!(r_ij, box_sizes)
            F_ij = force(r_ij)

            F_i += F_ij
            F_j[tid, j] -= F_ij #this needs to be reduced across warp at end
        end
    end

    #Reduce force's calculated by each thread in warp
    #*not quite right, probably just do a scan on each row??
    for i in [16,8,4,2,1] #idk how to write a loop that does this so just hard code for now
        F_j += shfl_down_sync(-1, F_j, i)
    end

    #Write piece of force for this tile in global mem
    tile_forces_i[tile_idx, tid, :] = F_i
    for j in tile.j_index_range #*move into loop where this got accumulated probably
        tile_forces_j[tile_idx, j, :] = 0.0 #& get value from reduced matrix
    end
end

function diagonal_tile_kernel(r, i_offset, j_offset, box_sizes, tid::Int32, 
        tile_forces_i::CuArray{Float32, 3}, tile_forces_j::CuArray{Float32, 3}, force::Function)

    for j in tid:(tid + WARP_SIZE) #*is THREADIDX.X 1 iNdexed???
        wrapped_j_idx = (j - 1) & (ATOM_BLOCK_SIZE - 1) #equivalent to modulo when ATOM_BLOCK_SIZE is power of 2
        if wrapped_j_idx < tid #Avoid double counting, causes warp divergence
            r_ij = r[i_offset + tid] .- r[j_offset + wrapped_j_idx]
            nearest_mirror!(r_ij, box_sizes)
            F_ij = force(r_ij)

            #No race conditions as threads in warp execute in step
            tile_forces_i[tile_idx, tid, :] += F_ij
            tile_forces_j[tile_idx, wrapped_j_idx, :] -= F_ij
        end
    end
end

# Each tile is assigned a warp of threads
# 1 tile per thread-block --> 1 Warp per block
function force_kernel(tile_forces_i::CuArray{Float32, 3}, tile_forces_j::CuArray{Float32, 3}, r::CuArray{Float32, 2},
    atom_flags::CuArray{Bool, 2}, force_function::Function, interaction_threshold::Int32)

    tile_idx = (blockIdx().x - 1i32)

    tile = tiles[tile_idx]

    #Overall index
    atom_i_idx = tile.i_index_range.start + threadIdx().x

    #Each thread loads its own atom data and the 32 atoms it is responble for into SHMEM
    atom_data_i = CuStaticSharedArray(Float32, (ATOM_BLOCK_SIZE, 3))
    atom_data_j = CuStaticSharedArray(Float32, (ATOM_BLOCK_SIZE, 3))


    atom_data_i[threadIdx().x,:] = r[atom_i_idx]
    for j in tile.j_index_range 
        atom_data_j[threadIdx().x,:] = r[j] 
    end

    __syncthreads()

    #This is gonna be the same for every thread in a warp
    #Wasted computation? could compute outside the kernel
    n_interactions = 0
    for j in tile.j_index_range
        n_interactions += atom_flags[tile.i, j]
    end

    #* Still need to check tile_interactions to see if we even bother calculating force

    if is_diagonal(tile)
        diagonal_tile_kernel(r, tile.i_index_range.start, tile.j_index_range.start,
            box_sizes, threadIdx().x, tile_forces_i, tile_forces_j, force_function)
    elseif n_interactions <= interaction_threshold
        partial_tile_kernel(r, box_sizes, threadIdx.x, tile_forces_i, tile_forces_j, force_function)
    else 
        full_tile_kernel(r, tile.i_index_range.start, tile.j_index_range.start,
             box_sizes, threadIdx().x, tile_forces_i, tile_forces_j, force_function)
    end

    return tile_forces
end

function calculate_force!(tnl::TiledNeighborList, sys::System, force_function::Function, forces::Vector,
     r_cut, r_skin, sort_atoms::Bool, check_box_interactions::Bool;
     interaction_threshold = Int32(12))

    if sort_atoms
        assign_atoms_to_voxels!(tnl, sys)
        sys, tnl = spatially_sort_atoms!(sys, tnl)
        tnl = build_bounding_boxes!(tnl, sys)
    end

    if check_box_interactions
        tnl = find_interacting_tiles!(tnl, sys, r_cut, r_skin)
    end

    #Launch CUDA kernel #& pre-allocate all these things outside of loop
    r = CuArray{Float32}(sys.atoms.positions)
    atom_flags = CuArray{Bool}(tnl.atom_flags)
    tile_forces_i_GPU = CUDA.zeros(Float32, (N_tiles, WARP_SIZE, 3))
    tile_forces_j_GPU = CUDA.zeros(Float32, (N_tiles, WARP_SIZE, 3))
    N_tiles = length(tnl.tile_interactions)

    threads_per_block = WARP_SIZE
    @cuda threads=threads_per_block blocks=N_tiles tile_forces =
         force_kernel!(tile_forces, r, atom_flags, force_function, interaction_threshold)

    #Copy forces back to CPU and reduce
    tile_forces_i_CPU = zeros(Float32, (N_tiles, WARP_SIZE, 3))
    tile_forces_j_CPU = zeros(Float32, (N_tiles, WARP_SIZE, 3))
    copyto!(tile_forces_i_CPU, tile_forces_i_GPU)
    copyto!(tile_forces_j_CPU, tile_forces_j_GPU)

    Threads.@threads for tile_idx in 1:N_tiles
        forces[tnl.tiles[tile_idx].i_index_range] .+= tile_forces_i_CPU[tile_idx]
    end

    Threads.@threads for tile_idx in 1:N_tiles
        forces[tnl.tiles[tile_idx].j_index_range] .+= tile_forces_j_CPU[tile_idx]
    end


    return tnl, forces
end