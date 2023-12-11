using Pkg
# using Revise
Pkg.activate(raw"C:\Users\ejmei\Repositories\CUDA_P3M\LongRangeInteractions")
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
ϵ = 0.1; σ = 3.492
potential = (r) -> LJ(r, ϵ, σ)
L = 16.86
r_cut_lj = 7.0
r_cut_real = 10.0
r_skin = 3.0
N_atoms = length(charges)

#Loop through configurations tested in LAMMPS and compare to our implementation
# for (i, positions) in enumerate(eachrow(trajectory))

#JUST RUN ONE SET OF POSITIONS
positions = trajectory[:,1]

#Stores data so that positions are contiguous in memory etc. 
atoms = StructArray{Atom}(position=positions, mass=masses, charge=charges, index=collect(1:N_atoms))
sys = System(atoms, L)

#Build neighbor list
voxel_width = get_optimal_voxel_width(r_cut, box_sizes)
tnl = TiledNeighborList(voxel_width, n_atoms(sys))
interacting_tiles = Tiles[]
forces = zeros(Float32, n_atoms(sys), 3)
energies = zeros(Float32, n_atoms(sys), 3)

# #Run SPME on system
# @timeit timer "SPME Loop $(i)" calculate_force!(tnl, sys, interacting_tiles,
#     potential, forces, energies, r_cut, r_skin, true, true)
        
# end

#Save timing data to file
timing_file_handle = open(joinpath(@__DIR__, "p3m_timings.txt"))
print_timer(timing_file_handle, timer, sortby=:firstexec)