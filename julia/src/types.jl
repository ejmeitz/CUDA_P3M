#High-level types, all specific interaction types are in the methods folder

abstract type LongRangeInteraction end

struct Atom{P,M,C}
    position::SVector{3,P} #only support 3D systems for now
    mass::M
    charge::C
    index::Int
end

function Atom(position, mass, index, charge = 0.0u"q")
    return Atom{eltype(position), typeof(mass), typeof(charge)}(
        position, mass, charge, index
    )
end

struct System{L} #Make <: AbstractSystem{3} in future
    atoms::StructArray{Atom}
    lattice_vec::Vector{Vector{L}}
end

function System(atoms, L)
    total_charge = sum(atoms.charge)
    @warn  total_charge == 0 "System must be charge neutral, total charge was $(total_charge)"

    lattice_vec = [[L,0,0],[0,L,0],[0,0,L]]

    return System{typeof(L)}(atoms, lattice_vec)
end

function System(atoms, lattice_vec::Vector{Vector{L}}) where {L}
    total_charge = sum(atoms.charge)
    @warn  total_charge == 0 "System must be charge neutral, total charge was $(total_charge)"

    return System{L}(atoms, lattice_vec)
end

positions(s::System) = s.atoms.position
positions(s::System, i::Integer) = s.atoms.position[i]
positions(s::System, slice::UnitRange{Int}) = view(s.atoms.position, slice, :)
positions(s::System, slice::UnitRange{Int}, i::Integer) = view(s.atoms.position, slice, i)


masses(s::System) = s.atoms.mass
masses(s::System, i::Integer) = s.atoms.mass[i]

charges(s::System) = s.atoms.charge
charges(s::System, i::Integer) = s.atoms.charge[i]
total_charge(s::System) = sum(charges(s))

n_atoms(s::System) = length(s.atoms)
lattice_vec(s::System) = s.lattice_vec
vol(s::System) = vol(lattice_vec(s))

#########################################################


abstract type TargetDevice{N} end

struct SingleThread <: TargetDevice{1} end #Mostly for benchmarking

struct CPU{N} <: TargetDevice{N} end #N corresponds to threads, for now single CPU

struct SingleGPU <: TargetDevice{1}
    device_id::Integer
end

struct MultiGPU{N} <: TargetDevice{N} #N corresponds to number of GPUs
    device_ids::SVector{N,Integer}
end

n_proc(::TargetDevice{N}) where {N} = N

####################################################

