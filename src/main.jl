include("init.jl")

#& Double check these are being parsed correctly
# Lammps can only give you coulombic energies, not forces
# will need to calculate the LJ part of the force manually and subtract.
trajectory, charges, masses, lammps_coul_energies, lammps_forces_all = load_test_data()

const timer = TimerOutput()

logger_output = joinpath(@__DIR__, "logs")

pp = PP()
pm = PM()
p3m = P3M(pp, pm)

lammps_errors = []
bruteforce_errors = []

#Loop through configurations tested in LAMMPS and compare to our implementation
for (i, positions) in enumerate(eachol(trajectory))
    
    #Stores data so that positions are contiguous in memory etc. 
    atoms = StructArray{Atom}(position = positions, mass = masses, charge = charges)

    #Run P3M on system
    @timeit timer "P3M Loop $(i)" coul_energies, coul_forces = p3m(atoms)
    log_step(joinpath(logger_output, "P3M_Step$(i)"), coul_energies, coul_forces)

    #Run self-coded force/energy loop, should brute force the coulombic term
    force_noncoul, bf_force_coul, bf_energy_coul = MD_loop(atoms)
    log_step(joinpath(logger_output, "BruteForce_Step$(i)"), bf_energy_coul, bf_force_coul)

    lammps_forces_coul = lammps_forces_all[:,i] .- force_noncoul

    err_lammps = compare_to_lammps(lammps_forces_coul, lammps_coul_energies[:,i], coul_forces, coul_energies)
    err_bf = compare_to_bruteforce(bf_force_coul, bf_energy_coul, coul_forces, coul_energies)
    push!(lammps_errors, err_lammps); push!(bruteforce_errors, err_bf)

end

#Save timing data to file
timing_file_handle = open(joinpath(@__DIR__, "p3m_timings.txt"))
print_timer(timing_file_handle, timer, sortby = :firstexec)