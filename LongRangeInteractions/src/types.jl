#High-level types, all specific interaction types are in the methods folder
export Atom, System, SPME, SingleThread, CPU, SingleGPU, MultiGPU,
    n_atoms, positions, n_mesh

abstract type LongRangeInteraction end

mutable struct Atom{M,C}
    const mass::M
    const charge::C
    id::Int
end

function Atom(mass, id, charge = 0.0u"q")
    return Atom{typeof(mass), typeof(charge)}(
        mass, charge, id
    )
end

struct System{P,L} #Make <: AbstractSystem{3} in future
    atoms::StructArray{Atom}
    # positions::Matrix{P}
    positions::Vector{Vector{P}}
    lattice_vec::Vector{Vector{L}}
end

function System(atoms, positions, L)
    total_charge = sum(atoms.charge)
    if total_charge != 0 
        @warn "System must be charge neutral, total charge was $(total_charge)"
    end
    lattice_vec = [[L,0,0],[0,L,0],[0,0,L]]

    return System{eltype(positions[1]),typeof(L)}(atoms, positions, lattice_vec)
end

function System(atoms, positions, lattice_vec::Vector{Vector{L}}) where {L}
    total_charge = sum(atoms.charge)
    if total_charge != 0 
        @warn "System must be charge neutral, total charge was $(total_charge)"
    end
    return System{eltype(positions),L}(atoms, positions, lattice_vec)
end

positions(s::System) = s.positions
positions(s::System, i::Integer) = s.positions[i]
positions(s::System, slice::UnitRange{<:Integer}) = view(s.positions, slice)
# positions(s::System, slice::UnitRange{<:Integer}, i::Integer) = view(s.positions, slice, i)


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

struct SPME{TD, T, R, B, E, L} <: LongRangeInteraction
    sys::System
    target_device::TD
    error_tol::T
    r_cut_dir::R
    β::B
    self_energy::E
    K::SVector{3,Integer}
    spline_order::Integer
    recip_lat::Vector{Vector{L}}
end

function SPME(sys, target_device, error_tol, r_cut_dir, spline_order)
    β = sqrt(-log(2*error_tol))/r_cut_dir
    self_energy = -(β/sqrt(π))*sum(x -> x*x, charges(sys))
    box_sizes = norm.(lattice_vec(sys))
    K = ceil.(Int, 2*β.*box_sizes./(3*(error_tol ^ 0.2)))

    recip_lat = reciprocal_vecs(sys.lattice_vec)

    return SPME{typeof(target_device), typeof(error_tol),
             typeof(r_cut_dir), typeof(β), typeof(self_energy), eltype(recip_lat[1])}(
                sys, target_device, error_tol, r_cut_dir, β, self_energy, K,
                spline_order, recip_lat)
end

reciprocal_lattice(spme::SPME) = spme.recip_lat
self_energy(spme::SPME) = spme.self_energy
n_mesh(spme::SPME) = spme.K
spline_order(spme::SPME) = spme.spline_order