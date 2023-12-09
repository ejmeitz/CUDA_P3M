const global WARP_SIZE = 32
const global TILE_SIZE = WARP_SIZE

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

struct TiledNeighborList
    voxel_width::Float64
    n_tiles::Int
    atom_flags::Array{Bool, 3}
    tile_interactions::Matrix{Bool}
    bounding_boxes::Vector{BoundingBox}
end

function TiledNeighborList(voxel_width, N_atoms)
    N_tiles = ceil(Int, N_atoms / TILE_SIZE)
    atom_flags = Array{Bool,3}(undef, (N_tiles, N_atoms))
    tile_interactions = Matrix{Bool}(undef, (N_tiles, N_tiles)) #lower half of this matrix is ignored
    bounding_boxes = Vector{BoundingBox}(undef, (N_tiles,))
    return TiledNeighborList(voxel_width, N_tiles, atom_flags, tile_interactions, bounding_boxes)
end