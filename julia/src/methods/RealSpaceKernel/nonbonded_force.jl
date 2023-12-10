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


function full_tile_kernel(sys::System, tid::Int32, forces::CuArray{Float32, 3})
     #Start loop in each thread at a different place
    #* does this cause warp divergence?
    for j in tid:(tid + WARP_SIZE) #*is THREADIDX.X 1 iNdexed???
        wrapped_j_idx = (j - 1) & (ATOM_BLOCK_SIZE - 1) #equivalent to modulo when ATOM_BLOCK_SIZE is power of 2

        r_ij = 
        nearest_mirror!(r_ij, )
        F_ij = force()

        #No race conditions as threads in warp execute in step
        forces[tile_idx, atom_i_idx, :] += F_ij
        forces[tile_idx, wrapped_j_idx, :] -= F_ij
    end
end

function partial_tile_kernel()
    F_i = 0.0f32
    #Store forces by pairs and reduce after block executes
    F_j = CuStaticSharedArray(Float32, (ATOM_BLOCK_SIZE, ATOM_BLOCK_SIZE, 3))

    
    for j in tile.j_index_range
        if tnl.atom_flags[tile.i, j]
            F_ij = force()
            F_i += F_ij
            F_j[threadIdx().x, j] -= F_ij #this needs to be reduced across warp at end
        end
    end

    #Reduce force's calculated by each thread in warp
    #*not quite right, probably just do a scan on each row??
    for i in [16,8,4,2,1] #idk how to write a loop that does this so just hard code for now
        F_j += shfl_down_sync(-1, F_j, i)
    end

    #Write piece of force for this tile in global mem
    forces[tile_idx, atom_i_idx, :] = F_i
    for j in tile.j_index_range #*move into loop where this got accumulated probably
        forces[tile_idx, j, :] = 0.0 #& get value from reduced matrix
    end
end

function diagonal_tile_kernel()

end

# Each tile is assigned a warp of threads
# 1 tile per thread-block --> 1 Warp per block
    #Could update to have multiple tiles per block
    #Could be a bit faster since less moves of data from main memory

function force_kernel(sys::System, tnl::TiledNeighborList, interaction_threshold::Int32)

    tile_idx = (blockIdx().x - 1i32)

    tile = tiles[tile_idx]

    #Overall index
    atom_i_idx = tile.i_index_range.start + threadIdx().x

    #Each thread loads its own atom data and the 32 atoms it is responble for into SHMEM
    atom_data_i = CuStaticSharedArray(Float32, (ATOM_BLOCK_SIZE, 3))
    atom_data_j = CuStaticSharedArray(Float32, (ATOM_BLOCK_SIZE, 3))


    atom_data_i[threadIdx().x,:] = positions(sys, atom_i_idx)
    for j in tile.j_index_range 
        atom_data_j[threadIdx().x,:] = positions(sys, j) 
    end
    

    __syncthreads()

    #This is gonna be the same for every thread in a warp
    #Wasted computation?
    n_interactions = 0
    for j in tile.j_index_range
        n_interactions += tnl.atom_flags[tile.i, j]
    end

    if is_diagonal(tile)
        diagonal_tile_kernel()
    elseif n_interactions <= interaction_threshold
        partial_tile_kernel()
    else 
        full_tile_kernel()
    end

    return nothing
end

#& Build some kind of neighbor list object
function calculate_force!(voxel_assignments, tnl::TiledNeighborList,
     sys::System, r_cut, r_skin, sort_atoms::Bool, check_box_interactions::Bool;
     interaction_threshold = 12)

    if sort_atoms
        assign_atoms_to_voxels!(voxel_assignments, sys, tnl.voxel_width)
        sys = spatially_sort_atoms(sys, voxel_assignments)
    end

    tnl = build_bounding_boxes!(tnl, sys)

    if check_box_interactions
        tnl = find_interacting_tiles!(tnl, sys, r_cut, r_skin)
    end

    #Launch CUDA kernel #TODO
    @cuda force_kernel()
end