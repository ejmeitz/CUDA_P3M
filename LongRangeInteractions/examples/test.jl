
function load_test_data(test_data_path)
    ld = LammpsDump(test_data_path)
    file = open(test_data_path, "r")

    N_atoms = ld.header_data["N_atoms"]
    N_samples = ld.n_samples
    N_cols = 10  #data columns: id m q x y z fx fy fz c_4

    data = zeros(N_atoms, N_cols)

    trajectory = zeros(N_atoms, 3, N_samples)
    forces_all = zeros(N_atoms, 3, N_samples)
    coul_energies = zeros(N_atoms, N_samples)
    charges = zeros(N_atoms)
    masses = zeros(N_atoms)

    for i in 1:N_samples

        parse_next_timestep!(data, ld, file, collect(1:N_cols))

        #Makes copies but this isnt in the timed section
        trajectory[:,:,i] .= data[:,4:6]
        forces_all[:,:,i] .= data[:,7:9]
        coul_energies[:,i] .= data[:,N_cols]
        masses .= data[:,3]
        charges .= data[:,2]
    end

    return trajectory, charges, masses, coul_energies, forces_all
end


function apply_pbc!(r::Vector{Vector{Float64}}, L)
    for i in eachindex(r)
        if r[i][1] < 0 || r[i][1] > L
            r[i][1] = r[i][1] - sign(r[i][1])*L
        end
        if r[i][2] < 0 || r[i][2] > L
            r[i][2] = r[i][2] - sign(r[i][2])*L
        end
        if r[i][3] < 0 || r[i][3] > L
            r[i][3] = r[i][3] - sign(r[i][3])*L
        end

    end
    return r

end

function compare_to_lammps(lammps_forces, lammps_energies, my_forces, my_energies)

end

function compare_to_bruteforce(bf_forces, bf_energies, my_forces, my_energies)

end

function log_step(filename, forces, energies)

end

#Runs MD to manually calculate teh energy/force of the water system simulated in LAMMPS
#Should brute force the Coulombic term to get a "ground truth" for quantifying error
function MD_loop()

end