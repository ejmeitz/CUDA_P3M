using Pkg
using Revise
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
using LongRangeInteractions
using TimerOutputs
using DataFrames
using StructArrays

include("test.jl")
include("dump_parser.jl")

#Initialize System from LAMMPS Data
test_data_path = joinpath(@__DIR__, "..", "test","test_data", "salt_sim_simple","dump.atom")
trajectory, charges, masses, lammps_coul_energies, lammps_forces_all = load_test_data(test_data_path)

const timer = TimerOutput()
logger_output = joinpath(@__DIR__, "logs")

#In salt_sim_simple σ and ϵ are same for all interactions
const ϵ = 0.1; const σ = 3.492
potential = (r) -> LJ(r, ϵ, σ)
L = 16.86
r_cut_lj = 7.0
r_cut_real = 10.0
r_skin = 3.0
N_atoms = length(charges)

#Loop through configurations tested in LAMMPS and compare to our implementation
# for (i, positions) in enumerate(eachrow(trajectory))

#JUST RUN ONE SET OF POSITIONS
positions = trajectory[:,:,1]

#Stores data so that positions are contiguous in memory etc. 
atoms = StructArray{Atom}(mass=masses, charge=charges, index=collect(1:N_atoms))
sys = System(atoms, positions, L)

#Build neighbor list
voxel_width = get_optimal_voxel_width(r_cut_lj, [L,L,L])
tnl = TiledNeighborList(voxel_width, n_atoms(sys));
interacting_tiles = Tile[]
forces = zeros(Float32, n_atoms(sys), 3);
energies = zeros(Float32, n_atoms(sys), 3);

# #Run SPME on system
i = 1
@timeit timer "SPME Loop $(i)" calculate_force!(tnl, sys, interacting_tiles,
    potential, forces, energies, [L,L,L], r_cut_lj, r_skin, true, true)
        
# end

#Save timing data to file
timing_file_handle = open(joinpath(@__DIR__, "p3m_timings.txt"))
print_timer(timing_file_handle, timer, sortby=:firstexec)