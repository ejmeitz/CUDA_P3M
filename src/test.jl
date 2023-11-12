
function load_test_data()
    test_data_path = joinpath(@__DIR__, "..", "test_data", "dump.atom")

    ld = LammpsDump(test_data_path)
    file = open(test_data_path, "r")

    N_atoms = ld.header_data["N_atoms"]
    N_samples = ld.n_samples
    N_cols = 10  #data columns: id m q x y z fx fy fz c_4

    data = zeros(N_atoms, N_cols)

    trajectory = zeros(3*N_atoms, N_samples)
    forces_all = zeros(3*N_atoms, N_samples)
    coul_energies = zeros(N_atoms, N_samples)
    charges = zeros(N_atoms)
    masses = zeros(N_atoms)

    for i in 1:N_samples

        parse_next_timestep!(data, ld, file, collect(1:N_cols))

        #Makes copies but this isnt in the timed section
        trajectory[:,i] .= reduce(vcat,data[:,4:6])
        forces_all[:,i] .= reduce(vcat,data[:,7:9])
        coul_energies[:,i] .= data[:,N_cols]
        masses .= data[:,2]
        charges .= data[:,3]
    end

    return trajectory, charges, masses, coul_energies, forces_all
end

function compare_to_lammps(lammps_forces, lammps_energies, my_forces, my_energies)

end

function compare_to_bruteforce(bf_forces, bf_energies, my_forces, my_energies)

end

function log_step(filename, forces, energies)

end