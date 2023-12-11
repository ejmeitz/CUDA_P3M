const global WARP_SIZE = 32
#kernel assumes ATOM_BLOCK_SIZE is a power of 2
const global ATOM_BLOCK_SIZE = WARP_SIZE #this is not refering to CUDA block size
const global TILE_SIZE = ATOM_BLOCK_SIZE*ATOM_BLOCK_SIZE

#Tile representing interaction between block i and j
struct Tile
    i::Int32
    j::Int32
    i_index_range::UnitRange{Int32} #Atom indexes of block i
    j_index_range::UnitRange{Int32} #Atom indexes of block j
end

is_diagonal(t::Tile) = t.i == t.j

struct BoundingBox
    xmin::FLoat64
    xmax::FLoat64
    ymin::FLoat64
    ymax::FLoat64
    zmin::FLoat64
    zmax::FLoat64
end

#Gives 0 distance if pt inside box
function boxPointDistance(bb::BoundingBox, pt)
    dx = max(max(bb.xmin - pt[1], pt[1] - bb.xmax), 0.0f);
    dy = max(max(bb.ymin - pt[2], pt[2] - bb.ymax), 0.0f);
    dz = max(max(bb.zmin - pt[3], pt[3] - bb.zmax), 0.0f);
    return sqrt(dx * dx + dy * dy + dz * dz);
end

function boxBoxDistance(bb1::BoundingBox, bb2::BoundingBox)

end

function get_block_idx_range(idx, N_atoms)
    lower_idx = (idx - 1)*TILE_SIZE + 1
    upper_idx = (idx - 1)*TILE_SIZE + TILE_SIZE
    upper_idx = upper_idx > N_atoms ? N_atoms : upper_idx
    return lower_idx, upper_idx
end


struct TiledNeighborList
    voxel_width::Float64
    n_blocks::Int
    voxel_assignments::Matrix{Int}
    atom_flags::Array{Bool, 3}
    tile_interactions::Vector{Bool}
    bounding_boxes::Vector{BoundingBox}
    tiles::Vector{Tiles}
end
"""
Parameters
- voxel_width : The width of the voxel used to sort atoms spatially. It is optimal
    if the number of voxels in each dimension is a power of 2. Typically, `voxel_width`
    is less than or the same as the cutoff radius.
"""
function TiledNeighborList(voxel_width, N_atoms)
    N_blocks = ceil(Int, N_atoms / ATOM_BLOCK_SIZE)
    # Interactions between block_i and atoms
    atom_flags = Array{Bool,3}(undef, (N_blocks, N_atoms))
    bounding_boxes = Vector{BoundingBox}(undef, (N_blocks,))
    voxel_assignments = zeros(N_atoms, 3)

    #Calcualte number of tiles on or above diagonal
    N_unique_tiles = Int64(0.5*N_blocks*(N_blocks + 1))
    tile_interactions = Vector{Bool}(undef, (N_unique_tiles,))

    #Pre-calcualte tile index ranges
    tiles = Vector{Tile}(undef, (N_unique_tiles,))
    for i in UnitRange{Int32}(1:N_blocks)
        lower_idx_i, upper_idx_i = get_block_idx_range(i, N_atoms)
        for j in UnitRange{Int32}(i:N_blocks)
            lower_idx_j, upper_idx_j = get_block_idx_range(j, N_atoms)
            tiles[i] = Tile(i, j, UnitRange{Int32}(lower_idx_i:upper_idx_i),
                 UnitRange{Int32}(lower_idx_j:upper_idx_j))
        end
    end
    
    tile_interactions = Vector{Bool}(undef, (N_unique_tiles,))

    return TiledNeighborList(voxel_width, N_blocks, voxel_assignments,
                atom_flags, tile_interactions, bounding_boxes, tiles)
end


##### Functions for Updating Neighbor List #####

"""
Assigns atoms into voxels. 
**ASSUMES ORIGIN IS (0,0,0) and SYSTEM IS CUBIC/RECTANGULAR**

Parameters:
- voxel_assignments: Nx3 matrix of voxel assignments which are the 3D
    index of the voxel each atom is inside of in 3D space
- sys::System: System object,
- voxel_width: Voxel is cude with side length `voxel_width`. 
    Ideally, this is chosen so that the number of voxels on 
    each side is a power of 2.

"""
function assign_atoms_to_voxels!(tnl::TiledNeighborList, sys::System)
    
    N_atoms = n_atoms(sys)
    for i in 1:N_atoms
        tnl.voxel_assignments[i,:] .= floor.(Int, positions(sys, i) ./ tnl.voxel_width) + 1
    end
    return tnl
end

# This only needs to be called once for NVT simulations
# Also assumes origin is (0,0,0)
function spatially_sort_voxels(n_voxels_per_dim::Vector)
    N_bits = ceil.(Int, log2.(n_voxels_per_dim))
    map_index = (i,j,k) -> [floor(Int64, ((i-1)/n_voxels_per_dim[1])*(2^N_bits[1])) + 1,
                          floor(Int64, ((j-1)/n_voxels_per_dim[2])*(2^N_bits[2])) + 1,
                          floor(Int64, ((k-1)/n_voxels_per_dim[2])*(2^N_bits[2])) + 1]

    #Integer indices of each voxel
    unmapped_indices = [[i,j,k] for i in 1:n_voxels_per_dim[1] for j in 1:n_voxels_per_dim[2] for k in 1:n_voxels_per_dim[3]]
    #Convert to numbers between 0 and 2^(N_bits)
    mapped_indices = [map_index(i,j,k) for i in 1:n_voxels_per_dim[1] for j in 1:n_voxels_per_dim[2] for k in 1:n_voxels_per_dim[3]]
    sorted_idxs = sortperm(mapped_indices, by = x ->  encode_hilbert(Compact(Int, N_bits), x))

    #return unmapped_indices[sorted_idxs]
    return Dict(unmapped_indices[sorted_idx] => sorted_idx for sorted_idx in sorted_idxs)
end

function spatially_sort_atoms!(sys::System, tnl::TiledNeighborList)

    n_voxels_per_dim = ceil.(Int, norm.(sys.lattice_vec) ./ tnl.voxel_width)
    tnl = assign_atoms_to_voxels!(tnl, sys)
    voxels_sorted = spatially_sort_voxels(n_voxels_per_dim)

    sort!(sys.atoms, by = atom -> tnl.voxel_assignments[voxels_sorted[atom.index]])

    return sys, tnl
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