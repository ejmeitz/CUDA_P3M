export get_optimal_voxel_width

function get_N_cells(lattice_vecs, r_cut)
    axb = cross(lattice_vecs[:, 1], lattice_vecs[:, 2])
    axb /= norm(axb)
    
    axc = cross(lattice_vecs[:, 1], lattice_vecs[:, 3])
    axc /= norm(axc)
    
    bxc = cross(lattice_vecs[:, 2], lattice_vecs[:, 3])
    bxc /= norm(bxc)

    proj = zeros(3)
    proj[1] = np.dot(lattice_vecs[:, 1], bxc)
    proj[2] = np.dot(lattice_vecs[:, 2], axc)
    proj[3] = np.dot(lattice_vecs[:, 3], axb)

    n = ceil.(Int64, r_cut / abs.(proj))

    return n

end

function vol(lat_vecs::Vector{Vector})
    return dot(lat_vecs[0], cross(lat_vecs[1], lat_vecs[2]))
end

function reciprocal_vecs(lat_vecs::Vector{Vector})
    V = vol(lat_vecs)
    m1 = cross(lat_vecs[2], lat_vecs[3])/V
    m2 = cross(lat_vecs[3], lat_vecs[1])/V
    m3 = cross(lat_vecs[1], lat_vecs[2])/V
    return [m1,m2,m3]
end

function reciprocal_vecs_twopi(lat_vecs)
    V = vol(lat_vecs)
    m1 = 2*π*cross(lat_vecs[2], lat_vecs[3])/V
    m2 = 2*π*cross(lat_vecs[3], lat_vecs[1])/V
    m3 = 2*π*cross(lat_vecs[1], lat_vecs[2])/V
    return [m1,m2,m3]
end

#ASSUMES CUBIC BOX
function get_optimal_voxel_width(r_cut, box_sizes)

    N_voxels = box_sizes ./ r_cut

    #Round down to nearest power of 2
    # N_voxels_down = Int(2^(floor(log2(N_voxels))))
    # voxel_width_down = box_sizes ./ N_voxels_down

    #Round up to nearest power of 2 -- want voxel_width <≈ r_cut
    N_voxels_up = Int.(2 .^(ceil.(log2.(N_voxels))))
    voxel_width_up = box_sizes ./ N_voxels_up


    return voxel_width_up
end