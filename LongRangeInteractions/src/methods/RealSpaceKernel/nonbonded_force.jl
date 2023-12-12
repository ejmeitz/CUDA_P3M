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
    F = -4*ϵ*(12*(k*k/r) + 6*(k/r))
    return F
end

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
    tile_forces_i, tile_forces_j, tile_energies_i, potential, r_ij, F_ij)
     #Start loop in each thread at a different place
    #* does this cause warp divergence?
    for j in tid:(tid + WARP_SIZE - 1)
        wrapped_j_idx = ((j - 1) & (ATOM_BLOCK_SIZE - 1)) + 1 #equivalent to modulo when ATOM_BLOCK_SIZE is power of 2

        r_ij .= r[i_offset + tid - 1] .- r[j_offset + wrapped_j_idx - 1]
        nearest_mirror!(r_ij, box_sizes)

        dist_ij = CUDA.norm3df(r_ij[1], r_ij[2], r_ij[3])

        #*figure out how to abstract this
        U_ij = LJ_potential(dist_ij, 0.1f32, 3.492f32)
        F_mag = LJ_force(dist_ij, 0.f321, 3.492f32)

        # #Convert F to be directional
        F_ij .= F_mag .* (r_ij ./ dist_ij)

        #No race conditions as threads in warp execute in step
        tile_energies_i[blockIdx().x, tid] += U_ij
        for d in 1:3
            tile_forces_i[blockIdx().x, tid, d] += F_ij[d]
            tile_forces_j[blockIdx().x, wrapped_j_idx, d] -= F_ij[d]
        end
    end
end

@inline function partial_tile_kernel(r, tile, atom_flags, box_sizes, tid::Int32, tile_forces_i,
         tile_forces_j, tile_energies_i, potential, r_ij, F_ij)

    F_ix = 0.0f32
    F_iy = 0.0f32
    F_iz = 0.0f32

    F_jx = 0.0f32
    F_jy = 0.0f32
    F_jz = 0.0f32
    
    count = 0
    for j in tile.j_index_range
        count += 1
        if atom_flags[tile.idx_1D, j]
            #Reset force tracker for atom j
            F_jx = 0.0f32
            F_jy = 0.0f32
            F_jz = 0.0f32

            r_ij .= r[tile.i_index_range.start + tid - 1] .- r[j]
            nearest_mirror!(r_ij, box_sizes)

            dist_ij = CUDA.norm3df(r_ij[1], r_ij[2], r_ij[3])

            #*figure out how to abstract this
            U_ij = LJ_potential(dist_ij, 0.1f32, 3.492f32)
            F_mag = LJ_force(dist_ij, 0.f321, 3.492f32)

            # #Convert F to be directional
            F_ij .= F_mag .* (r_ij ./ dist_ij)

            F_ix += F_ij[1]; F_iy += F_ij[2]; F_iz += F_ij[3]
            F_jx -= F_ij[1]; F_jy -= F_ij[2]; F_jz -= F_ij[3]

            tile_energies_i[blockIdx().x, tid] += U_ij

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

            #*test function doesnt work with f32??
            # function reduce_kernel()
            #     F_j = 1.1f32
            #     i = 16i32
            #     while i > 0i32
            #         F_j += CUDA.shfl_down_sync(0xFFFFFFFF, F_j, i)
            #         i >>= 1i32
            #     end
            #     @cuprintln "$(F_j + 1)"
            #     return nothing
            # end
            
        end
    end

    tile_forces_i[blockIdx().x, tid, 1] = F_ix
    tile_forces_i[blockIdx().x, tid, 2] = F_iy
    tile_forces_i[blockIdx().x, tid, 3] = F_iz

end


@inline function diagonal_tile_kernel(r, i_offset, j_offset, box_sizes, tid::Int32, 
        tile_forces_i, tile_forces_j, tile_energies_i, potential, r_ij, F_ij)

    for j in tid:(tid + WARP_SIZE - 1)
        wrapped_j_idx = ((j - 1) & (ATOM_BLOCK_SIZE - 1)) + 1#equivalent to modulo when ATOM_BLOCK_SIZE is power of 2
        if wrapped_j_idx < tid #Avoid double counting, causes warp divergence
            r_ij .= r[i_offset + tid - 1] .- r[j_offset + wrapped_j_idx - 1]
            nearest_mirror!(r_ij, box_sizes) #*causes divergence??
            dist_ij = CUDA.norm3df(r_ij[1], r_ij[2], r_ij[3])

            #*figure out how to abstract this
            U_ij = LJ_potential(dist_ij, 0.1f32, 3.492f32)
            F_mag = LJ_force(dist_ij, 0.f321, 3.492f32)

            # #Convert F to be directional
            F_ij .= F_mag .* (r_ij ./ dist_ij)

            # #No race conditions as threads in warp execute in step
            tile_energies_i[blockIdx().x, tid] += U_ij
            for d in 1:3
                tile_forces_i[blockIdx().x, tid, d] += F_ij[d]
                tile_forces_j[blockIdx().x, wrapped_j_idx, d] -= F_ij[d]
            end

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
            n_interactions += atom_flags[tile.idx_1D, j]
        end

        #* Still need to check tile_interactions to see if we even bother calculating force

        if is_diagonal(tile)
            diagonal_tile_kernel(r, tile.i_index_range.start, tile.j_index_range.start,
                box_sizes, threadIdx().x, tile_forces_i, tile_forces_j, tile_energies_i, potential,
                r_ij, F_ij)
        elseif n_interactions <= interaction_threshold
            partial_tile_kernel(r, tile, atom_flags, box_sizes, threadIdx().x, tile_forces_i, tile_forces_j, 
                tile_energies_i, potential, r_ij, F_ij)
        else
            full_tile_kernel(r, tile.i_index_range.start, tile.j_index_range.start,
                 box_sizes, threadIdx().x, tile_forces_i, tile_forces_j, 
                tile_energies_i, potential,r_ij, F_ij)
        end
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
    tile_forces_i_CPU = Array(tile_forces_i_GPU)
    tile_forces_j_CPU = Array(tile_forces_j_GPU)
    tile_energies_i_CPU = Array(tile_energies_i_GPU)

    # Threads.@threads for tile_idx in 1:N_tiles_interacting
    #     forces[tnl.tiles[tile_idx].i_index_range] .+= tile_forces_i_CPU[tile_idx]
    #     energies[tnl.tiles[tile_idx].i_index_range] .+= tile_energies_i_CPU[tile_idx]
    # end

    # Threads.@threads for tile_idx in 1:N_tiles_interacting
    #     forces[tnl.tiles[tile_idx].j_index_range] .+= tile_forces_j_CPU[tile_idx]
    # end


    return tnl, forces
end