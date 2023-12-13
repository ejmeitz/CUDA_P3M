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

# const ϵ = 0.1; const σ = 3.492
@inline function LJ_potential(r::Float32, ϵ::Float32, σ::Float32)
    k = (σ/r)^6
    return 4*ϵ*(k*(k-1))
end

@inline function LJ_force(r::Float32, ϵ::Float32, σ::Float32)
    k = (σ/r)^6
    F = 4*ϵ*(12*(k*k/r) - 6*(k/r))
    return F
end

#&how u do this without warp divergence??
@inline function nearest_mirror!(r, L)
    if r > L/2
        r -= L
    elseif r < -L/2
        r += L
    end
    
    return r
end

@inline function full_tile_kernel(r, i_offset, j_offset, box_sizes, tid::Int32, 
    tile_forces_i, tile_forces_j, tile_energies_i, potential)

    r_ijx = 0.0f0; r_ijy = 0.0f0; r_ijz = 0.0f0
    F_ij_x = 0.0f0; F_ij_y = 0.0f0; F_ij_z = 0.0f0
    U_tot = 0.0f0

     #Start loop in each thread at a different place
    #* does this cause warp divergence?
    for j in tid:(tid + WARP_SIZE - 1)
        wrapped_j_idx = ((j - 1) & (ATOM_BLOCK_SIZE - 1)) + 1 #equivalent to modulo when ATOM_BLOCK_SIZE is power of 2

        r_ijx = r[i_offset + tid - 1, 1] - r[j_offset + wrapped_j_idx - 1, 1]
        r_ijy = r[i_offset + tid - 1, 2] - r[j_offset + wrapped_j_idx - 1, 2]
        r_ijz = r[i_offset + tid - 1, 3] - r[j_offset + wrapped_j_idx - 1, 3]

        nearest_mirror!(r_ijx, box_sizes[1])
        nearest_mirror!(r_ijy, box_sizes[2])
        nearest_mirror!(r_ijz, box_sizes[3])

        dist_ij = CUDA.norm3df(r_ijx, r_ijy, r_ijz)

        #*figure out how to abstract this
        U_ij = LJ_potential(dist_ij, 0.1f0, 3.492f0)
        F_mag = LJ_force(dist_ij, 0.1f0, 3.492f0)

        # #Convert F to be directional
        F_ij_x = F_mag * (r_ijx / dist_ij)
        F_ij_y = F_mag * (r_ijy / dist_ij)
        F_ij_z = F_mag * (r_ijz / dist_ij)

        U_tot += U_ij

        #No race conditions as threads in warp execute in step
        tile_forces_i[blockIdx().x, tid, 1] += F_ij_x
        tile_forces_i[blockIdx().x, tid, 2] += F_ij_y
        tile_forces_i[blockIdx().x, tid, 3] += F_ij_z
        tile_forces_j[blockIdx().x, wrapped_j_idx, 1] -= F_ij_x
        tile_forces_j[blockIdx().x, wrapped_j_idx, 2] -= F_ij_y
        tile_forces_j[blockIdx().x, wrapped_j_idx, 3] -= F_ij_z
    end
    #*could move forces_i out here too, to minimize writes
    tile_energies_i[blockIdx().x, tid] = U_tot

end

@inline function partial_tile_kernel(r, tile, atom_flags, box_sizes, tid::Int32, tile_forces_i,
         tile_forces_j, tile_energies_i, potential)

    F_ix = 0.0f0; F_iy = 0.0f0; F_iz = 0.0f0
    F_jx = 0.0f0; F_jy = 0.0f0; F_jz = 0.0f0
    F_ij_x = 0.0f0; F_ij_y = 0.0f0; F_ij_z = 0.0f0
    r_ijx = 0.0f0; r_ijy = 0.0f0; r_ijz = 0.0f0
    U_tot = 0.0f0
    
    count = 0
    for j in tile.j_index_range
        count += 1
        if atom_flags[tile.idx_1D, j]
            #Reset force tracker for atom j
            F_jx = 0.0f0; F_jy = 0.0f0; F_jz = 0.0f0

            r_ijx = r[tile.i_index_range.start + tid - 1, 1] - r[j, 1]
            r_ijy = r[tile.i_index_range.start + tid - 1, 2] - r[j, 2]
            r_ijz = r[tile.i_index_range.start + tid - 1, 3] - r[j, 3]

            nearest_mirror!(r_ijx, box_sizes[1])
            nearest_mirror!(r_ijy, box_sizes[2])
            nearest_mirror!(r_ijz, box_sizes[3])

            dist_ij = CUDA.norm3df(r_ijx, r_ijy, r_ijz)

            U_ij = LJ_potential(dist_ij, 0.1f0, 3.492f0)
            F_mag = LJ_force(dist_ij, 0.1f0, 3.492f0)

            # #Convert F to be directional
            F_ij_x = F_mag * (r_ijx / dist_ij)
            F_ij_y = F_mag * (r_ijy / dist_ij)
            F_ij_z = F_mag * (r_ijz / dist_ij)

            F_ix += F_ij_x; F_iy += F_ij_y; F_iz += F_ij_z
            F_jx -= F_ij_x; F_jy -= F_ij_y; F_jz -= F_ij_z
            U_tot += U_ij

            # sync_threads() #probably not needed since warp in step

            #Reduce F_j across warp
            idx = 16i32
            while idx > 0i32
                F_jx += shfl_down_sync(0xFFFFFFFF, F_jx, idx)
                F_jy += shfl_down_sync(0xFFFFFFFF, F_jy, idx)
                F_jz += shfl_down_sync(0xFFFFFFFF, F_jz, idx)
                idx >>= 1i32
            end

            #& is it bad to have all the threads write same vals to same address?
            tile_forces_j[blockIdx().x, count, 1] = F_jx
            tile_forces_j[blockIdx().x, count, 2] = F_jy
            tile_forces_j[blockIdx().x, count, 3] = F_jz            
        end
    end

    #*might be worht moving inside the loop to save registers, writes probably go async
    tile_energies_i[blockIdx().x, tid] = U_tot
    tile_forces_i[blockIdx().x, tid, 1] = F_ix
    tile_forces_i[blockIdx().x, tid, 2] = F_iy
    tile_forces_i[blockIdx().x, tid, 3] = F_iz

end


@inline function diagonal_tile_kernel(r, i_offset, j_offset, box_sizes, tid::Int32, 
        tile_forces_i, tile_forces_j, tile_energies_i, potential)

    r_ijx = 0.0f0; r_ijy = 0.0f0; r_ijz = 0.0f0
    F_ij_x = 0.0f0; F_ij_y = 0.0f0; F_ij_z = 0.0f0

    for j in tid:(tid + WARP_SIZE - 1)
        wrapped_j_idx = ((j - 1) & (ATOM_BLOCK_SIZE - 1)) + 1#equivalent to modulo when ATOM_BLOCK_SIZE is power of 2
        if wrapped_j_idx < tid #Avoid double counting, causes warp divergence

            r_ijx = r[i_offset + tid - 1, 1] - r[j_offset + wrapped_j_idx - 1, 1]
            r_ijy = r[i_offset + tid - 1, 2] - r[j_offset + wrapped_j_idx - 1, 2]
            r_ijz = r[i_offset + tid - 1, 3] - r[j_offset + wrapped_j_idx - 1, 3]

            #* causes divergence??
            nearest_mirror!(r_ijx, box_sizes[1])
            nearest_mirror!(r_ijy, box_sizes[2])
            nearest_mirror!(r_ijz, box_sizes[3])

            dist_ij = CUDA.norm3df(r_ijx, r_ijy, r_ijz)

            #*figure out how to abstract this
            U_ij = LJ_potential(dist_ij, 0.1f0, 3.492f0)
            F_mag = LJ_force(dist_ij, 0.1f0, 3.492f0)

            # #Convert F to be directional
            F_ij_x = F_mag * (r_ijx / dist_ij)
            F_ij_y = F_mag * (r_ijy / dist_ij)
            F_ij_z = F_mag * (r_ijz / dist_ij)

            # #No race conditions as threads in warp execute in step
            tile_energies_i[blockIdx().x, tid] += U_ij
            tile_forces_i[blockIdx().x, tid, 1] += F_ij_x
            tile_forces_i[blockIdx().x, tid, 2] += F_ij_y
            tile_forces_i[blockIdx().x, tid, 3] += F_ij_z
            tile_forces_j[blockIdx().x, wrapped_j_idx, 1] -= F_ij_x
            tile_forces_j[blockIdx().x, wrapped_j_idx, 2] -= F_ij_y
            tile_forces_j[blockIdx().x, wrapped_j_idx, 3] -= F_ij_z

        end
    end
end

# Each tile is assigned a warp of threads
# 1 tile per thread-block --> 1 Warp per block
function force_kernel!(tile_forces_i, tile_forces_j, tile_energies_i, tiles, 
    r, box_sizes, atom_flags, potential::Function, interaction_threshold::Int32)

    CUDA.Const(atom_flags) #*not sure this does much
    CUDA.Const(tiles)
    CUDA.Const(box_sizes)

    tile = tiles[blockIdx().x]
    atom_i_idx = tile.i_index_range.start + (threadIdx().x - 1)
    
    if atom_i_idx <= size(r)[1]

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
            n_interactions += atom_flags[tile.idx_1D, j]
        end

        #* HOW TO CHECK IF J IS IN BOUNDS INSIDE EACH KERNEL?
        if is_diagonal(tile)
            diagonal_tile_kernel(r, tile.i_index_range.start, tile.j_index_range.start,
                box_sizes, threadIdx().x, tile_forces_i, tile_forces_j, tile_energies_i, potential)
        elseif n_interactions <= interaction_threshold
            partial_tile_kernel(r, tile, atom_flags, box_sizes, threadIdx().x, tile_forces_i, tile_forces_j, 
                tile_energies_i, potential)
        else
            full_tile_kernel(r, tile.i_index_range.start, tile.j_index_range.start,
                 box_sizes, threadIdx().x, tile_forces_i, tile_forces_j, 
                tile_energies_i, potential)
        end
    end

    return nothing

end

#box_sizes assumes rectangular box
function calculate_force!(tnl::TiledNeighborList, sys::System, interacting_tiles::Vector{Tile},
     potential::Function, forces::Matrix, energies::Vector, box_sizes, r_cut, r_skin,
     update_neighbor_list::Bool; interaction_threshold = Int32(12))

    if update_neighbor_list
        sys, tnl = spatially_sort_atoms!(sys, tnl)
        tnl = build_bounding_boxes!(tnl, sys)
        tnl = find_interacting_tiles!(tnl, sys, r_cut, r_skin)
        interacting_tiles = tnl.tiles[tnl.tile_interactions] #allocation
    end

    N_tiles_interacting = length(interacting_tiles)
    print("$(N_tiles_interacting) tiles interacting\n")

    #Launch CUDA kernel
    r = CuArray{Float32}(reduce(hcat, positions(sys))') #*does this make another alloc?
    atom_flags = CuArray{Bool}(tnl.atom_flags)
    cu_interacting_tiles = CuArray{Tile}(interacting_tiles)
    tile_forces_i_GPU = CUDA.zeros(Float32, (N_tiles_interacting, WARP_SIZE, 3))
    tile_forces_j_GPU = CUDA.zeros(Float32, (N_tiles_interacting, WARP_SIZE, 3))
    tile_energies_i_GPU = CUDA.zeros(Float32, (N_tiles_interacting, WARP_SIZE))

    threads_per_block = WARP_SIZE
    @cuda threads=threads_per_block blocks=N_tiles_interacting force_kernel!(tile_forces_i_GPU, tile_forces_j_GPU,
         tile_energies_i_GPU, cu_interacting_tiles, r, CuArray{Float32}(box_sizes), atom_flags, potential,
          interaction_threshold)


    println("==============")
    println("KERNEL COMPLETED")
    println("==============")

    #Copy forces and energy back to CPU and reduce
    tile_forces_i_CPU = Array(tile_forces_i_GPU)
    tile_forces_j_CPU = Array(tile_forces_j_GPU)
    tile_energies_i_CPU = Array(tile_energies_i_GPU)
    println(sum(tile_energies_i_CPU))
    Threads.@threads for tile_idx in 1:N_tiles_interacting
        len = length(interacting_tiles[tile_idx].i_index_range) #last tile doesnt have same length
        forces[interacting_tiles[tile_idx].i_index_range,:] .+= tile_forces_i_CPU[tile_idx,1:len,:]
        energies[interacting_tiles[tile_idx].i_index_range] .+= tile_energies_i_CPU[tile_idx,1:len]

    end

    Threads.@threads for tile_idx in 1:N_tiles_interacting
        len = length(interacting_tiles[tile_idx].j_index_range)
        forces[interacting_tiles[tile_idx].j_index_range,:] .+= tile_forces_j_CPU[tile_idx,1:len,:]
    end


    return tnl, forces
end



# function reduce_kernel()
#     F_j = 1.1f0
#     i = 16i32
#     while i > 0i32
#         F_j += CUDA.shfl_down_sync(0xFFFFFFFF, F_j, i)
#         i >>= 1i32
#     end
#     @cuprintln "$(F_j + 1)"
#     return nothing
# end