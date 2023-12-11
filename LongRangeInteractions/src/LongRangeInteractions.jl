module LongRangeInteractions

using LinearAlgebra
using StructArrays
using StaticArrays
using TimerOutputs
using Unitful
using BijectiveHilbert 
# using Adapt

using CUDA #this should be conditionally loaded if TargetDevice is a GPU

include("helper.jl")
include("types.jl")

include("./methods/RealSpaceKernel/tiled_neighbor_list.jl")
include("./methods/RealSpaceKernel/nonbonded_force.jl")

include("./methods/SPME/splines.jl")
include("./methods/SPME/SPME.jl")
include("./methods/SPME/interpolate_charge.jl")
include("./methods/SPME/particle_particle.jl")
include("./methods/SPME/particle_mesh.jl")

end