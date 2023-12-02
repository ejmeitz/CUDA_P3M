
abstract type TargetDevice{N} end

struct CPU{N} <: TargetDevice{N} end #N corresponds to threads, for now single CPU

struct GPU{N} <: TargetDevice{N} #N corresponds to number of GPUs
    device_ids::SVector{N,Integer}
end 

n_proc(::TargetDevice{N}) where N = N

####################################################

abstract type LongRangeInteraction end


struct SPME{TD} <: LongRangeInteraction
    target_device::TD
end


####################################################

struct Atom{P,M,C}
    position::SVector{3,P} #only support 3D systems for now
    mass::M
    charge::C
end

function Atom(position, mass, charge = 0.0u"q")
    return Atom{eltype(position), typeof(mass), typeof(charge)}(
        position, mass, charge
    )
end

struct System #Make <: AbstractSystem{3} in future
    atoms::StructArray{Atom}
end

function System(atoms)
    total_charge = sum(atoms.charge)
    @assert  total_charge == 0 "System must be charge neutral, total charge was $(total_charge)"

    return System(atoms)
end

positions(s::System) = s.atoms.position
positions(s::System, i::Integer) = s.atoms.position[i]

masses(s::System) = s.atoms.mass
masses(s::System, i::Integer) = s.atoms.mass[i]

charges(s::System) = s.atoms.charge
charges(s::System, i::Integer) = s.atoms.charge[i]
total_charge(s::System) = sum(charges(s))

n_atoms(s::System) = length(s.atoms)