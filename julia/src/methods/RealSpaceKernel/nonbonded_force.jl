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

function get_tile_idx_range(tile_idx, N_atoms)
    lower_idx = (tile_idx - 1)*TILE_SIZE + 1
    upper_idx = (tile_idx - 1)*TILE_SIZE + TILE_SIZE
    upper_idx = upper_idx > N_atoms ? N_atoms : upper_idx
    return lower_idx, upper_idx
end

function build_bounding_boxes!(tnl::TiledNeighborList, sys::System)

    N_atoms = n_atoms(sys)
    for tile_idx in 1:tnl.n_tiles
        lower_idx, upper_idx = get_tile_idx_range(tile_idx, N_atoms)

        tnl.bounding_boxes[tile_idx] = BoundingBox(
            min(positions(sys, lower_idx:upper_idx, 1)), max(positions(sys, lower_idx:upper_idx, 1)),
            min(positions(sys, lower_idx:upper_idx, 2)), max(positions(sys, lower_idx:upper_idx, 2)),
            min(positions(sys, lower_idx:upper_idx, 3)), max(positions(sys, lower_idx:upper_idx, 3))
        )
    end

    return tnl

end

#Checks if atoms in tile_j are within r_cut of bounding box of tile_i
function set_interaction_flags!(tnl::TiledNeighborList, sys::System, tile_i, tile_j, r_cut)

    N_atoms = n_atoms(sys)
    lower_idx, upper_idx = get_tile_idx_range(tile_j, N_atoms)

    for (j, atom_j) in enuemrate(eachrow(positions(sys, lower_idx:upper_idx)))
        tnl.atom_flags[tile_i, j] =  (boxPointDistance(tnl.bounding_boxes[tile_i], atom_j) < r_cut)
    end

    return tnl
end

"""
tile_interactions is N_Tiles x N_Tiles upper triangular matrix
atom_flags is a N_tiles x N_atoms x N_atoms where the last two dims are upper triangular matrix #*this feels like more memory than necessary
"""
function find_interacting_tiles!(tnl::TiledNeighborList, sys::System, r_cut, r_skin)

    for tile_i in 1:tnl.N_tiles
        #Set self interaction to true for blocks on diagonal
        tnl.tile_interactions[tile_i,tile_i] = true
        lower_idx, upper_idx = get_tile_idx_range(tile_i, N_atoms)
        tnl.atom_flags[tile_i, lower_idx:upper_idx] .= true

        for tile_j in (i+1):tnl.N_tiles
            @views tnl.tile_interactions[tile_i, tile_j] = 
                (boxBoxDistance(bounding_box_dims[tile_i,:], bounding_box_dims[tile_j,:]) < r_cut + r_skin)
            
            #If two tile interact
            if tile_interactions[tile_i,tile_j] == true
                tnl.atom_flags = set_interaction_flags!(tnl, sys, tile_i, tile_j, r_cut)
            end
        end
    end
    
    return tile_interactions, atom_flags

end


function full_tile_kernel(sys::System)

    #N_tiles = blockDim().x

    tile_idx = blockIdx().x
    local_atom_idx = threadIdx().x
    global_thd_idx = ((tile_idx - 1i32) * blockDim().x) + local_atom_idx

    #Each thread loads its own atom data and the 32 atoms it is responble for into SHMEM
    atom_data_i = CuStaticSharedArray(::Float32, (TILE_SIZE,3))
    atom_data_j = CuStaticSharedArray(::Float32, (TILE_SIZE,3))

    atom_data_i[local_atom_idx] = positions(sys, global_thd_idx)

    for j in 1:TILE_SIZE
        atom_data_j[j] = positions(sys, ) #* TODO
    end

    __syncthreads()
    
    #Calculate interaction with all other 32 atoms in tile j
    for j in 1:TILE_SIZE
        
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

    #Compute interactions
    for tile_i in 1:tnl.N_Tiles
        
        for tile_j in range(tile_i, N_tiles)
            lower_idx_j, upper_idx_j = get_tile_idx_range(tile_j, N_atoms)
            n_interactions = sum(tnl.atom_flags[tile_i, lower_idx:upper_idx_j])

            if n_interactions <= interaction_threshold #This is just from OpenMM paper, make a parameter
                partial_tile_kernel() #__device__ kernel
            else
                full_tile_kernel() #__device__ kernel
            end
        end
    end
end