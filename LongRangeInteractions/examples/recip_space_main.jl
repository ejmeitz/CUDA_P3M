using Pkg
using Revise
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.resolve();Pkg.instantiate()
using LongRangeInteractions
using TimerOutputs
using DataFrames
using BenchmarkTools
using StructArrays
using CUDA
using FFTW

include("test.jl")
include("dump_parser.jl")

#Initialize System from LAMMPS Data
test_data_path = joinpath(@__DIR__, "..", "test","test_data", "salt_sim_simple_4UC","dump.atom")
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

err_tol = 1e-5
n = 6
spme = SPME(sys, SingleThread(), err_tol, r_cut_real, n);

#Pre-allocate Q and dQdr and u
Q = zeros(n_mesh(spme)...);
dQdr = zeros(N_atoms, 3, n_mesh(spme)...); #* this is a huge array with fine meshes
u = [Vector{Float64}(undef, (length(n_mesh(spme)), )) for _ in eachindex(positions)];

BC = calc_BC(spme);
# @btime interpolate_charge!(u, Q, dQdr, spme); #& this one way faster but I broke something

# Q2 = zeros(n_mesh(spme)...);
interpolate_charge2!(Q, dQdr, spme) #* gives slightly different results than Python??

@benchmark CUDA.@sync begin

    BC_cuda = CuArray{Float32}(BC)
    Q_cuda = CuArray{Float32}(Q)
    Q_inv = fft(Q_cuda)
    Q_inv .*= BC_cuda
    Q_conv_theta = ifft(Q_inv)

    A = 332.0637132991921
    E = 0.5 * A* sum(real(Q_conv_theta) .* real(Q_cuda))
end


#Save timing data to file
# timing_file_handle = open(joinpath(@__DIR__, "p3m_timings.txt"))
# print_timer(timing_file_handle, timer, sortby=:firstexec)