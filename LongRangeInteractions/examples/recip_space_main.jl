using Pkg
using Revise
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.resolve();Pkg.instantiate()
using LongRangeInteractions
using TimerOutputs
using DataFrames
using BenchmarkTools
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
r_cut_real = 7.0
r_skin = 3.0
N_atoms = length(charges)

#Loop through configurations tested in LAMMPS and compare to our implementation
# for (i, positions) in enumerate(eachrow(trajectory))

#JUST RUN ONE SET OF POSITIONS
positions = trajectory[:,:,1]
positions = [positions[i,:] for i in 1:size(positions)[1]]
positions = apply_pbc!(positions, L)

#Stores data so that positions are contiguous in memory etc. 
atoms = StructArray{Atom}(mass=masses, charge=charges, id=collect(1:N_atoms));
sys = System(atoms, positions, L);

err_tol = 1e-4
n = 5
spme = SPME(sys, SingleThread(), err_tol, r_cut_real, n);

#Pre-allocate Q and dQdr
Q = zeros(n_mesh(spme)...);
dQdr = zeros(N_atoms, 3, n_mesh(spme)...);

BC = calc_BC(spme)


BC_cuda = CuArray{Float32}(BC)
Q_cuda = CuArray{Float32}(Q)
Q_inv = fft(Q_cuda)
Q_inv .= BC_cuda
Q_conv_theta = ifft(Q_inv)

E = 0.5 * sum(real(Q_conv_theta) .* real(Q_cuda))





return E


@benchmark interpolate_charge!($Q, $dQdr, $spme)


#Save timing data to file
# timing_file_handle = open(joinpath(@__DIR__, "p3m_timings.txt"))
# print_timer(timing_file_handle, timer, sortby=:firstexec)