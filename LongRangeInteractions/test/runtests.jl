using Test
using DelimitedFiles

include("dump_parser.jl")
include("LJ_energy_loop.jl")


@testset "LJ Energy Matches LAMMPS -- Salt 3UC" begin

    #Load Data From Dump File
    salt_data = joinpath(@__DIR__, "test_data", "salt_sim", "dump.atom")
    ld = LammpsDump(salt_data)
    dump_file = open(ld.path, "r")

    #Load LJ Energy
    energy_data = joinpath(@__DIR__, "test_data", "salt_sim", "energy_breakdown.txt")
    U_LJ_LAMMPS = readdlm(energy_data, " ", comments = true)[3,:]

    #Make 
    box_sizes = [ld.header_data["L_x"], ld.header_data["L_y"], ld.header_data["L_z"]]
    r_cut_lj = 7.0u"Å"

    for step in 1:ld.n_samples
        parse_next_timestep!(ld, dump_file)

        positions = ld.data_storage[!,["x", "y", "z"]]
        forces = ld.data_storage[!,["fx", "fy", "fz"]]
        charges = ld.data_storage[!, "q"]

        #Only compare energies here, we only know total force
        U_LJ, _ = lj_energy_loop(positions, charges, box_sizes, r_cut_lj)
        
        #This should match to machine precision basically
        @test U_LJ ≈ U_LJ_LAMMPS[step] atol = 1e-7
    end

    close(dump_file)
end

@testset "CPU Ewald Matches LAMMPS -- Salt 3UC" begin

end