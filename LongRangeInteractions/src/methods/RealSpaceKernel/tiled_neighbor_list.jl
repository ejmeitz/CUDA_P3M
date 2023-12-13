export TiledNeighborList, Tile

const global WARP_SIZE::Int64 = 32
#kernel assumes ATOM_BLOCK_SIZE is a power of 2
const global ATOM_BLOCK_SIZE::Int64 = WARP_SIZE #this is not refering to CUDA block size
const global INTERACTIONS_PER_TILE::Int64 = ATOM_BLOCK_SIZE*ATOM_BLOCK_SIZE

#Tile representing interaction between block i and j
struct Tile
    i::Int32
    j::Int32
    idx_1D::Int32
    i_index_range::UnitRange{Int32} #Atom indexes of block i
    j_index_range::UnitRange{Int32} #Atom indexes of block j
end

is_diagonal(t::Tile) = t.i == t.j

struct BoundingBox
    bb_min::NTuple{3,Float64}
    bb_max::NTuple{3,Float64}
end

xmin(bb::BoundingBox) = bb.bb_min[1]
ymin(bb::BoundingBox) = bb.bb_min[2]
zmin(bb::BoundingBox) = bb.bb_min[3]
xmax(bb::BoundingBox) = bb.bb_max[1]
ymax(bb::BoundingBox) = bb.bb_max[2]
zmax(bb::BoundingBox) = bb.bb_max[3]

#*remake these without sqrt?
#Gives 0 distance if pt inside box
function boxPointDistance(bb::BoundingBox, pt)
    dx = max(max(xmin(bb) - pt[1], pt[1] - xmax(bb)), 0.0);
    dy = max(max(ymin(bb) - pt[2], pt[2] - ymax(bb)), 0.0);
    dz = max(max(zmin(bb) - pt[3], pt[3] - zmax(bb)), 0.0);
    return sqrt(dx * dx + dy * dy + dz * dz);
end

# Assumes bounding boxes are cubic/rectangular
# Gives zero of boxes overlap
# https://stackoverflow.com/questions/65107289/minimum-distance-between-two-axis-aligned-boxes-in-n-dimensions
function boxBoxDistance(bb1::BoundingBox, bb2::BoundingBox)
    delta1 = bb1.bb_min .- bb2.bb_max
    delta2 = bb2.bb_min .- bb1.bb_max
    u = max.(zeros(length(delta1)), delta1)
    v = max.(zeros(length(delta2)), delta2)
    dist = norm([u; v])
    return dist
end

function get_block_idx_range(idx, N_atoms)
    lower_idx = (idx - 1)*ATOM_BLOCK_SIZE + 1
    upper_idx = (idx - 1)*ATOM_BLOCK_SIZE + ATOM_BLOCK_SIZE
    upper_idx = upper_idx > N_atoms ? N_atoms : upper_idx
    return lower_idx, upper_idx
end

struct TiledNeighborList
    voxel_width::Vector{Float64}
    n_blocks::Int
    voxel_assignments::Matrix{Int}
    atom_flags::Matrix{Bool}
    tile_interactions::Vector{Bool}
    bounding_boxes::Vector{BoundingBox}
    tiles::Vector{Tile}
end

#Allows this type to be constructed with CUDA.jl
# function Adapt.adapt_structure(to, tnl::TiledNeighborList)
#     voxel_assignments = Adapt.adapt_structure(to, tnl.voxel_assignments)
#     atom_flags = Adapt.adapt_structure(to, tnl.atom_flags)
#     tile_interactions = Adapt.adapt_structure(to, tnl.tile_interactions)
#     bounding_boxes = Adapt.adapt_structure(to, tnl.bounding_boxes)
#     tiles = Adapt.adapt_structure(to, tnl.tiles)
#     TiledNeighborList(tnl.voxel_width, tnl.n_blocks, voxel_assignments, atom_flags, tile_interactions, bounding_boxes, tiles)
# end


"""
Parameters
- voxel_width : The width of the voxel used to sort atoms spatially. It is optimal
    if the number of voxels in each dimension is a power of 2. Typically, `voxel_width`
    is less than or the same as the cutoff radius.
"""
function TiledNeighborList(voxel_width, N_atoms)
    N_blocks = ceil(Int, N_atoms / ATOM_BLOCK_SIZE)
    # Interactions between block_i and atoms

    bounding_boxes = Vector{BoundingBox}(undef, (N_blocks,))
    voxel_assignments = zeros(N_atoms, 3)

    #Calcualte number of tiles on or above diagonal
    N_unique_tiles = Int64(0.5*N_blocks*(N_blocks + 1))
    tile_interactions = Vector{Bool}(undef, (N_unique_tiles,))
    atom_flags = Matrix{Bool}(undef, (N_unique_tiles, N_atoms))

    #Pre-calcualte tile index ranges
    tiles = Vector{Tile}(undef, (N_unique_tiles,))
    idx_1D = 1
    for i in UnitRange{Int32}(1:N_blocks)
        lower_idx_i, upper_idx_i = Int32.(get_block_idx_range(i, N_atoms))
        for j in UnitRange{Int32}(i:N_blocks)
            lower_idx_j, upper_idx_j = Int32.(get_block_idx_range(j, N_atoms))
            tiles[idx_1D] = Tile(i, j, idx_1D, UnitRange{Int32}(lower_idx_i:upper_idx_i),
                 UnitRange{Int32}(lower_idx_j:upper_idx_j))
            idx_1D += 1
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
"""
function get_voxel_assignment(r, voxel_width)
    return floor.(Int, r ./ voxel_width) .+ 1
end

"""
Assigns atoms into voxels. 
**ASSUMES ORIGIN IS (0,0,0) and SYSTEM IS CUBIC/RECTANGULAR**
"""
function assign_atoms_to_voxels!(tnl::TiledNeighborList, sys::System)
    
    N_atoms = n_atoms(sys)
    for i in 1:N_atoms
        tnl.voxel_assignments[i,:] .= get_voxel_assignment(positions(sys, i), tnl.voxel_width)
    end
    return tnl
end

#This for debugging
function check_voxel_counts(tnl, sys)
    N_atoms = n_atoms(sys)
    counts = Dict()
    for i in 1:N_atoms
        voxel = tnl.voxel_assignments[sys.atoms[i].id,:]
        if haskey(counts, voxel)
            counts[voxel] += 1
        else
            counts[voxel] = 1
        end
    end
    return counts
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
    # print(voxels_sorted)
    #*double calculating voxel assignments rn
    sort!(sys.atoms, by = atom -> voxels_sorted[tnl.voxel_assignments[atom.id,:]])
    sort!(sys.positions, by = r -> voxels_sorted[get_voxel_assignment(r, tnl.voxel_width)])


    return sys, tnl
end


function build_bounding_boxes!(tnl::TiledNeighborList, sys::System)

    N_atoms = n_atoms(sys)

    for block_idx in 1:tnl.n_blocks
        lower_idx, upper_idx = get_block_idx_range(block_idx, N_atoms)

        xmin = minimum(getindex.(positions(sys, lower_idx:upper_idx), 1))
        ymin = minimum(getindex.(positions(sys, lower_idx:upper_idx), 2))
        zmin = minimum(getindex.(positions(sys, lower_idx:upper_idx), 3))
        xmax = maximum(getindex.(positions(sys, lower_idx:upper_idx), 1))
        ymax = maximum(getindex.(positions(sys, lower_idx:upper_idx), 2))
        zmax = maximum(getindex.(positions(sys, lower_idx:upper_idx), 3))

        tnl.bounding_boxes[block_idx] = BoundingBox((xmin, ymin, zmin), (xmax, ymax, zmax))
    end

    return tnl

end

#Checks if atoms in tile_j are within r_cut of bounding box of tile_i
function set_atom_flags!(tnl::TiledNeighborList, sys::System, tile::Tile, r_cut)

    #* THIS SHOULD BE NEAREST MIRROR ATOM PROBABLY
    for (j, atom_j) in enumerate(positions(sys, tile.j_index_range))
        tnl.atom_flags[tile.idx_1D, j] =  (boxPointDistance(tnl.bounding_boxes[tile.i], atom_j) < r_cut)
    end

    return tnl
end


function find_interacting_tiles!(tnl::TiledNeighborList, sys::System, r_cut, r_skin)

    for (t,tile) in enumerate(tnl.tiles)
        if is_diagonal(tile)
            tnl.tile_interactions[t] = true
            tnl.atom_flags[t, tile.j_index_range] .= true
        end

        #* THIS SHOULD BE NEAREST MIRROR ATOM PROBABLY
        tiles_interact = boxBoxDistance(tnl.bounding_boxes[tile.i], tnl.bounding_boxes[tile.j]) < r_cut + r_skin
        tnl.tile_interactions[t] = tiles_interact

        if tiles_interact
            tnl = set_atom_flags!(tnl, sys, tile, r_cut)
        end

    end

    return tnl
end