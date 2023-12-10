const global WARP_SIZE = 32
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
    atom_flags::Array{Bool, 3}
    tile_interactions::Vector{Bool}
    bounding_boxes::Vector{BoundingBox}
    tiles::Vector{Tiles}
end

function TiledNeighborList(voxel_width, N_atoms)
    N_blocks = ceil(Int, N_atoms / ATOM_BLOCK_SIZE)
    # Interactions between block_i and atoms
    atom_flags = Array{Bool,3}(undef, (N_blocks, N_atoms))
    bounding_boxes = Vector{BoundingBox}(undef, (N_blocks,))

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

    return TiledNeighborList(voxel_width, N_blocks, atom_flags, tile_interactions, bounding_boxes, tiles)
end