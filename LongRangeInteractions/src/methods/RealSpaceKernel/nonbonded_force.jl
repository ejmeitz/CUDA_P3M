export calculate_force!
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
@inline function nearest_mirror!(r_ij, box_sizes)
    for i in eachindex(r_ij)
        if r_ij[i] > box_sizes[i]/2
            r_ij[i] -= box_sizes[i]
        elseif r_ij[i] < -box_sizes[i]/2
            r_ij[i] += box_sizes[i]
        end
    end
    return r_ij
end

@inline function full_tile_kernel(r, i_offset, j_offset, box_sizes, tid::Int32, 
    tile_forces_i, tile_forces_j,
    tile_energies_i, potential)
     #Start loop in each thread at a different place
    #* does this cause warp divergence?
    for j in tid:(tid + WARP_SIZE) #*is THREADIDX.X 1 iNdexed???
        wrapped_j_idx = (j - 1) & (ATOM_BLOCK_SIZE - 1) #equivalent to modulo when ATOM_BLOCK_SIZE is power of 2

        r_ij = r[i_offset + tid] .- r[j_offset + wrapped_j_idx]
        nearest_mirror!(r_ij, box_sizes)
        U_ij, F_ij = potential(r_ij)

        #Convert F to be directional
        r_hat = 

        #No race conditions as threads in warp execute in step
        tile_energies_i[tile_idx, tid] += U_ij
        tile_forces_i[tile_idx, tid, :] += F_ij
        tile_forces_j[tile_idx, wrapped_j_idx, :] -= F_ij
    end
end

@inline function partial_tile_kernel(r, box_sizes, tid::Int32, tile_forces_i,
         tile_forces_j, tile_energies_i, potential)

    F_i = 0.0f32
    U_i = 0.0f32
    #Store forces by pairs and reduce after block executes
    F_j = CuStaticSharedArray(Float32, (ATOM_BLOCK_SIZE, ATOM_BLOCK_SIZE, 3))

    
    for j in tile.j_index_range
        if tnl.atom_flags[tile.i, j]
            r_ij = r[i] .- r[j]
            nearest_mirror!(r_ij, box_sizes)
            U_ij, F_ij = potential(r_ij)

            #Convert F to be directional
            r_hat = 

            F_i += F_ij
            U_i += U_ij
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
    tile_energies_i[tile_idx, tid] = U_i
    for j in tile.j_index_range #*move into loop where this got accumulated probably
        tile_forces_j[tile_idx, j, :] = 0.0 #& get value from reduced matrix
    end
end

@inline function diagonal_tile_kernel(r, i_offset, j_offset, box_sizes, tid::Int32, 
        tile_forces_i, tile_forces_j, tile_energies_i, potential, r_ij, F_ij)

    for j in tid:(tid + WARP_SIZE)
        wrapped_j_idx = (j - 1) & (ATOM_BLOCK_SIZE - 1) #equivalent to modulo when ATOM_BLOCK_SIZE is power of 2
        if wrapped_j_idx < tid #Avoid double counting, causes warp divergence
            r_ij .= r[i_offset + tid] .- r[j_offset + wrapped_j_idx]
            nearest_mirror!(r_ij, box_sizes) #*causes divergence??
            U_ij, F_mag = potential(r_ij) #*cant return a tuple like this

            # #Convert F to be directional
            F_ij .= F_mag .* (r_ij .* CUDA.rsqrt(r_ij[1]*r_ij[1] + r_ij[2]*r_ij[2] + r_ij[3]*r_ij[3]))
            # F_ij .= F_mag.*(r_ij*CUDA.rnorfmf(3,r_ij[1]))#* this is not exposed by CUDA.jl

            # #No race conditions as threads in warp execute in step
            # tile_energies_i[blockIdx().x, tid] += U_ij
            # tile_forces_i[blockIdx().x, tid, :] .+= F_ij
            # tile_forces_j[blockIdx().x, wrapped_j_idx, :] .-= F_ij
        end
    end
end

# Each tile is assigned a warp of threads
# 1 tile per thread-block --> 1 Warp per block
function force_kernel!(tile_forces_i, tile_forces_j, tile_energies_i, tiles, 
    r, box_sizes, atom_flags, potential::Function, interaction_threshold::Int32, r_ij, F_ij)

    # CUDA.Const(atom_flags) #*not sure this does much

    tile = tiles[blockIdx().x]
    atom_i_idx = tile.i_index_range.start + (threadIdx().x - 1)
    # @cuprintln("Tile Idx: $atom_i_idx")

    #* avoid out of bounds accesses on last tile or when tid is high...
    if atom_i_idx < size(r)[1]

        # Each thread loads its own atom data and the 32 atoms it is responble for into SHMEM
        atom_data_i = CuStaticSharedArray(Float32, (ATOM_BLOCK_SIZE, 3))
        atom_data_j = CuStaticSharedArray(Float32, (ATOM_BLOCK_SIZE, 3))


        atom_data_i[threadIdx().x,1] = r[atom_i_idx,1]
        atom_data_i[threadIdx().x,2] = r[atom_i_idx,2]
        atom_data_i[threadIdx().x,3] = r[atom_i_idx,3]
        for j in tile.j_index_range 
            for d in 1:3
                atom_data_j[threadIdx().x,d] = r[j,d] 
            end
        end

        sync_threads()

        #This is gonna be the same for every thread in a warp
        #Wasted computation? could compute outside the kernel
        n_interactions = 0
        for j in tile.j_index_range
            n_interactions += atom_flags[tile.i, j]
        end

        #* Still need to check tile_interactions to see if we even bother calculating force

        if is_diagonal(tile)
            diagonal_tile_kernel(r, tile.i_index_range.start, tile.j_index_range.start,
                box_sizes, threadIdx().x, tile_forces_i, tile_forces_j, tile_energies_i, potential,
                r_ij, F_ij)
        end
        # elseif n_interactions <= interaction_threshold
        #     partial_tile_kernel(r, box_sizes, threadIdx.x, tile_forces_i, tile_forces_j, 
        #         tile_energies_i, potential)
        # else 
        #     full_tile_kernel(r, tile.i_index_range.start, tile.j_index_range.start,
        #          box_sizes, threadIdx().x, tile_forces_i, tile_forces_j, tile_energies_i, potential)
        # end
    end

    return nothing

end

#box_sizes assumes rectangular box
function calculate_force!(tnl::TiledNeighborList, sys::System, interacting_tiles::Vector{Tile},
     potential::Function, forces::Matrix, energies::Matrix, box_sizes, r_cut, r_skin, sort_atoms::Bool,
      check_box_interactions::Bool; interaction_threshold = Int32(12))

    if sort_atoms
        sys, tnl = spatially_sort_atoms!(sys, tnl)
        tnl = build_bounding_boxes!(tnl, sys)
    end

    if check_box_interactions
        tnl = find_interacting_tiles!(tnl, sys, r_cut, r_skin)
        interacting_tiles = tnl.tiles[tnl.tile_interactions] #allocation
    end

    N_tiles_interacting = length(interacting_tiles)

    #Launch CUDA kernel #& pre-allocate all these things outside of loop
    r = CuArray{Float32}(positions(sys))
    atom_flags = CuArray{Bool}(tnl.atom_flags)
    cu_interacting_tiles = CuArray{Tile}(interacting_tiles)
    tile_forces_i_GPU = CUDA.zeros(Float32, (N_tiles_interacting, WARP_SIZE, 3))
    tile_forces_j_GPU = CUDA.zeros(Float32, (N_tiles_interacting, WARP_SIZE, 3))
    tile_energies_i_GPU = CUDA.zeros(Float32, (N_tiles_interacting, WARP_SIZE))

    #Storage for things in kernel
    r_ij = CUDA.zeros(Float32, 3)
    F_ij = CUDA.zeros(Float32, 3)

    threads_per_block = WARP_SIZE
    @cuda threads=threads_per_block blocks=N_tiles_interacting force_kernel!(tile_forces_i_GPU, tile_forces_j_GPU,
         tile_energies_i_GPU, cu_interacting_tiles, r, CuArray{Float32}(box_sizes), atom_flags, potential,
          interaction_threshold, r_ij, F_ij)


    println("==============")
    println("KERNEL COMPLETED")
    println("==============")

    #Copy forces and energy back to CPU and reduce
    # tile_forces_i_CPU = Array(tile_forces_i_GPU)
    # tile_forces_j_CPU = Array(tile_forces_j_GPU)
    # tile_energies_i_CPU = Array(tile_energies_i_GPU)

    # Threads.@threads for tile_idx in 1:N_tiles_interacting
    #     forces[tnl.tiles[tile_idx].i_index_range] .+= tile_forces_i_CPU[tile_idx]
    #     energies[tnl.tiles[tile_idx].i_index_range] .+= tile_energies_i_CPU[tile_idx]
    # end

    # Threads.@threads for tile_idx in 1:N_tiles_interacting
    #     forces[tnl.tiles[tile_idx].j_index_range] .+= tile_forces_j_CPU[tile_idx]
    # end


    return tnl, forces
end