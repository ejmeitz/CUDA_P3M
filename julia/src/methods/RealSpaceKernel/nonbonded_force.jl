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


function full_tile_kernel(sys::System, tnl::TiledNeighborList)
    #Calculate interaction of atom i with all other 32 atoms in tile j
    for j in 1:TILE_SIZE
        F_ij = force()
        forces[global_atom_idx,:] += F_ij

        #how tf u do this without data races??
        global_j_idx =  
        forces[local_atom_idx,:] -= F_ij
    end
end

function partial_tile_kernel()

end

function diagonal_tile_kernel()

end

# One block fo every pair of interacting tiles
function force_kernel(sys::System, tnl::TiledNeighborList)

    #N_tiles = blockDim().x

    tile_i = ceil(Int32, threadIdx().x / TILE_SIZE) + (TILES_PER_BLOCK * (blockIdx().x - 1i32))

    local_atom_idx = 

    global_atom_idx = ((blockIdx().x - 1i32) * blockDim().x) + threadIdx().x

    #Each thread loads its own atom data and the 32 atoms it is responble for into SHMEM
    atom_data_i = CuStaticSharedArray(Float32, (TILE_SIZE,3))
    atom_data_j = CuStaticSharedArray(Float32, (TILE_SIZE,3))

    atom_data_i[local_atom_idx,:] = positions(sys, global_thd_idx)
    atom_data_j[local_atom_idx,:] = positions(sys, ) #* TODO

    __syncthreads()


    for tile_j in 1:(tile_i-1)
        lower_idx_j, upper_idx_j = get_tile_idx_range(tile_j, N_atoms)
        n_interactions = sum(tnl.atom_flags[tile_i, lower_idx_j:upper_idx_j])

        if n_interactions <= interaction_threshold #This is just from OpenMM paper, make a parameter
            partial_tile_kernel() #__device__ kernel
        else
            full_tile_kernel() #__device__ kernel
        end
    end

    #tile_i == tile_j
    diagonal_tile_kernel()

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

    #Compute interactions
    for tile_idx in 1:length(tnl.tiles)
        

        lower_idx_j, upper_idx_j = get_tile_idx_range(tile_idx.second, N_atoms)
        n_interactions = sum(tnl.atom_flags[tile_i, lower_idx:upper_idx_j])


        # if n_interactions <= interaction_threshold #This is just from OpenMM paper, make a parameter
        #     partial_tile_kernel() #__device__ kernel
        # else
        #     full_tile_kernel() #__device__ kernel
        # end

        # diagonal_kernel() #__device__ kernel
    end
end