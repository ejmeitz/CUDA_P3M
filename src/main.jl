include("init.jl")

trajectory, charges, masses, lammps_energies, lammps_forces = load_test_data()

const timer = TimerOutput()

pp = PP()
pm = PM()
p3m = P3M(pp, pm)

#Loop through configurations tested in LAMMPS and compare to our implementation
for (i,positions) in enumerate(trajectory)
    
    atoms = StructArray{Atom}(position = positions, mass = masses, charge = charges)

    #Run P3M on system
    @timeit timer "P3M Loop $(i)" energies, forces = p3m(atoms)


    #Compare to ground truth
    ground_truth_energy = lammps_energies[i]
    ground_truth_force = lammps_forces[i]


end

#Save timing data to file
timing_file_handle = open(joinpath(@__DIR__, "p3m_timings.txt"))
print_timer(timing_file_handle, timer, sortby = :firstexec)