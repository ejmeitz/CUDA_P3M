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



"""
Assigns atoms into voxels. 
**ASSUMES ORIGIN IS (0,0,0) and SYSTEM IS CUBIC**

Parameters:
- voxel_assignments: Nx3 matrix of voxel assignments which are the 3D
    index of the voxel each atom is inside of in 3D space
- sys::System: System object,
- voxel_width: Voxel is cude with side length `voxel_width`

"""
function assign_atoms_to_voxels!(voxel_assignments::Matrix{Integer}, 
        sys::System, voxel_width) where T
    
    N_atoms = n_atoms(sys)
    for i in 1:N_atoms
        voxel_assignments[i,:] .= positions(sys, i) ./ voxel_width
    end
    return voxel_assignments
end

# This only needs to be called once for NVT simulations
function build_hilbert_mapping(n_voxels_per_dim)

end

function spatially_sort_atoms!(sys, voxel_assignments)

end


function build_bounding_boxes!(tnl::TiledNeighborList, sys::System)

    N_atoms = n_atoms(sys)

    for block_idx in 1:tnl.n_blocks
        lower_idx, upper_idx = get_block_idx_range(block_idx, N_atoms)

        tnl.bounding_boxes[block_idx] = BoundingBox(
            min(positions(sys, lower_idx:upper_idx, 1)), max(positions(sys, lower_idx:upper_idx, 1)),
            min(positions(sys, lower_idx:upper_idx, 2)), max(positions(sys, lower_idx:upper_idx, 2)),
            min(positions(sys, lower_idx:upper_idx, 3)), max(positions(sys, lower_idx:upper_idx, 3))
        )
    end

    return tnl

end

#Checks if atoms in tile_j are within r_cut of bounding box of tile_i
function set_atom_flags!(tnl::TiledNeighborList, sys::System, tile_i, tile_j, r_cut)

    N_atoms = n_atoms(sys)
    lower_idx, upper_idx = get_tile_idx_range(tile_j, N_atoms)


    #* THIS SHOULD BE NEAREST MIRROR ATOM PROBABLY
    for (j, atom_j) in enuemrate(eachrow(positions(sys, lower_idx:upper_idx)))
        tnl.atom_flags[tile_i, j] =  (boxPointDistance(tnl.bounding_boxes[tile_i], atom_j) < r_cut)
    end

    return tnl
end


function find_interacting_tiles!(tnl::TiledNeighborList, sys::System, r_cut, r_skin)

    for (t,tile) in enumerate(tnl.tiles)
        if is_diagonal(tile)
            tnl.tile_interactions[t] = true
            tnl.atom_flags[t, tile.j_index_range] .= true
        end

        tiles_interact = boxBoxDistance(tnl.bounding_boxes[tile.i], bounding_box_dims[tile.j, :]) < r_cut + r_skin
        tnl.tile_interactions[t] = tiles_interact

        if tiles_interact
            tnl = set_atom_flags!(tnl, sys, tile_i, tile_j, r_cut)
        end

    end

    return tnl
end


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
    else # calculate all interactions
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