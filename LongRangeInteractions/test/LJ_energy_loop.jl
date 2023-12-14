function nearest_mirror(r_ij,box_sizes)
    Lx,Ly,Lz = box_sizes
    r_x,r_y,r_z = r_ij
        
    if r_x > Lx/2
        r_x -= Lx
    else if r_x < -Lx/2
        r_x += Lx
    end
        
    if r_y > Ly/2
        r_y -= Ly
    else if r_y < -Ly/2
        r_y += Ly 
    end
        
    if r_z > Lz/2
        r_z -= Lz
    else if r_z < -Lz/2
        r_z += Lz
    end

    return [r_x,r_y,r_z]
end

function LJ(r, ep, sig)
    k = (sig/r)^6
    U = 4*ep*((k^2) - k)
    F = 4*ep*((12*(sig^12)/(r^13)) - (6*(sig^6)/(r^7)))
    return U, F
end

function LJ_NaNa(r)
    ep = 0.1
    sig = 2.583
    return LJ(r, ep, sig)
end

function LJ_ClCl(r)
    ep = 0.1
    sig = 4.401
    return LJ(r, ep, sig)
end

function LJ_NaCl(r)
    ep = 0.1
    sig = 3.492
    return LJ(r, ep, sig)
end

# To calculate LJ part of interaction
function lj_energy_loop(positions, charges, box_sizes, r_cut_real)
    N_atoms = len(charges)

    forces = zeros(N_atoms, 3)
    U = zeros(N_atoms)

    for i in range(1,N_atoms)
        for j in range(i+1, N_atoms)
            r_ij = positions[i] .- positions[j]
            r_ij = nearest_mirror(r_ij, box_sizes)

            dist_ij = norm(r_ij)

            if dist_ij < r_cut_real

                if charges[i] == 1.0 && charges[j] == 1.0 #both Na
                    U_ij, F_ij = LJ_NaNa(dist_ij)
                else if charges[i] == -1.0 && charges[j] == -1.0 #both Cl
                    U_ij, F_ij = LJ_ClCl(dist_ij)
                else #Na + Cl
                    U_ij, F_ij = LJ_NaCl(dist_ij)
                end

                r_hat = r_ij ./ dist_ij 
                F_ij = F_ij .* r_hat

                forces[i,:] .+= F_ij
                forces[j,:] .-= F_ij
                U[i] += U_ij
            end
        end
    end

    return U, forces
end