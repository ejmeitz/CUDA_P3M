using LongRangeInteractions
using TimerOutputs


#Initialize System from LAMMPS Data
trajectory, charges, masses, lammps_coul_energies, lammps_forces_all = load_test_data()

const timer = TimerOutput()
logger_output = joinpath(@__DIR__, "logs")

ϵ = 0.24037
σ = 3.4
potential = (r) -> LJ(r, ϵ, σ)
L = 16.86
r_cut = 
r_skin = 

#Loop through configurations tested in LAMMPS and compare to our implementation
for (i, positions) in enumerate(eachol(trajectory))

    #Stores data so that positions are contiguous in memory etc. 
    atoms = StructArray{Atom}(position=positions, mass=masses, charge=charges)
    sys = System(atoms, L)

    #Build neighbor list
    voxel_width = get_optimal_voxel_width(r_cut, box_sizes)
    tnl = TiledNeighborList(voxel_width, n_atoms(sys))
    interacting_tiles = Tiles[]
    forces = zeros(Float32, n_atoms(sys), 3)
    energies = zeros(Float32, n_atoms(sys), 3)

    #Run SPME on system
    @timeit timer "SPME Loop $(i)" calculate_force!(tnl, sys, interacting_tiles,
        potential, forces, energies, r_cut, r_skin, true, true)
        
end

#Save timing data to file
timing_file_handle = open(joinpath(@__DIR__, "p3m_timings.txt"))
print_timer(timing_file_handle, timer, sortby=:firstexec)