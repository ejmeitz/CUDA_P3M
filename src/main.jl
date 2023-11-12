include("init.jl")

#& Double check these are being parsed correctly
# Lammps can only give you coulombic energies, not forces
# will need to calculate the LJ part of the force manually and subtract.
trajectory, charges, masses, lammps_coul_energies, lammps_forces_all = load_test_data()

const timer = TimerOutput()

pp = PP()
pm = PM()
p3m = P3M(pp, pm)

#Loop through configurations tested in LAMMPS and compare to our implementation
for (i, positions) in enumerate(eachol(trajectory))
    
    atoms = StructArray{Atom}(position = positions, mass = masses, charge = charges)

    #Run P3M on system
    @timeit timer "P3M Loop $(i)" coul_energies, coul_forces = p3m(atoms)

    #####################
    # Compare to LAMMPS #
    #####################
    ground_truth_couleng = lammps_coul_energies[:,i]
    ground_truth_force_all = lammps_forces_all[:,i]

    #Calculate non-Coulombic part of forces
    ground_truth_force_noncoul = non_coulombic_force(positions)

    #Calculate Errors


    ######################################
    # Compare to Brute Force Calculation #
    ######################################
end

#Save timing data to file
timing_file_handle = open(joinpath(@__DIR__, "p3m_timings.txt"))
print_timer(timing_file_handle, timer, sortby = :firstexec)