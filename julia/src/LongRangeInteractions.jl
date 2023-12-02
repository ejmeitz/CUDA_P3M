module LongRangeInteractions

using LinearAlgebra
using StructArrays
using TimerOutputs

#include all files here
include("types.jl")
include("dump_parser.jl")
include("test.jl")

include("particle_particle.jl")
include("particle_mesh.jl")

end