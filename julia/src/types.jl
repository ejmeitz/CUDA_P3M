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

struct System{L} #Make <: AbstractSystem{3} in future
    atoms::StructArray{Atom}
    box_size::SVector{3, L}
end

function System(atoms, box_size)
    total_charge = sum(atoms.charge)
    @assert  total_charge == 0 "System must be charge neutral, total charge was $(total_charge)"

    return System{eltype(box_size)}(atoms, box_size)
end

positions(s::System) = s.atoms.position
positions(s::System, i::Integer) = s.atoms.position[i]

masses(s::System) = s.atoms.mass
masses(s::System, i::Integer) = s.atoms.mass[i]

charges(s::System) = s.atoms.charge
charges(s::System, i::Integer) = s.atoms.charge[i]
total_charge(s::System) = sum(charges(s))

n_atoms(s::System) = length(s.atoms)
box_size(s::System) = s.box_size

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

abstract type LongRangeInteraction end


struct SPME{TD, T, R, B, E} <: LongRangeInteraction
    sys::System
    target_device::TD
    error_tol::T
    r_cut_dir::R
    β::B
    self_energy::E
    K::SVector{3,Integer}
end

function SPME(sys, target_device, error_tol, r_cut_dir)
    β = sqrt(-log(2*error_tol))/r_cut_dir
    self_energy = -(β/sqrt(π))*sum(x -> x*x, charges(sys))
    K = ceil.(Int, 2*β*box_size(sys)/(3*(error_tol ^ 0.2)))

    return SPME{typeof(target_device), typeof(error_tol),
             typeof(r_cut_dir), typeof(β), typeof(self_energy)}(
                sys, target_device, error_tol, r_cut_dir, β, self_energy, K)
end

self_energy(spme::SPME) = spme.self_energy
n_mesh(spme::SPME) = spme.K

####################################################