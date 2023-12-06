struct SPME{TD, T, R, B, E, L} <: LongRangeInteraction
    sys::System
    target_device::TD
    error_tol::T
    r_cut_dir::R
    β::B
    self_energy::E
    K::SVector{3,Integer}
    spline_order::Integer
end

function SPME(sys, target_device, error_tol, r_cut_dir, spline_order)
    β = sqrt(-log(2*error_tol))/r_cut_dir
    self_energy = -(β/sqrt(π))*sum(x -> x*x, charges(sys))
    K = ceil.(Int, 2*β*box_size(sys)/(3*(error_tol ^ 0.2)))


    return SPME{typeof(target_device), typeof(error_tol),
             typeof(r_cut_dir), typeof(β), typeof(self_energy), eltype(recip_lat[1])}(
                sys, target_device, error_tol, r_cut_dir, β, self_energy, recip_lat, K,
                spline_order)
end

reciprocal_lattice(spme::SPME) = spme.recip_lat
self_energy(spme::SPME) = spme.self_energy
n_mesh(spme::SPME) = spme.K
spline_order(spme::SPME) = spme.spline_order

function run!(spme::SPME{SingleThread})

    #& could write to reuse storage for F
    E_dir, F_dir = particle_particle(spme)
    E_rec, F_rec = particle_mesh(spme)
    E_self = self_energy(spme)

    #* Need to convert units here, try not to hard code to specific unit system
    #* Probably best to just use Unitful and multiply E by 1/4πϵ₀

    E_SPME = E_dir + E_rec + E_self
    F_SPME = F_dir .+ F_rec

    return E_SPME, F_SPME

end

function run(spme::SPME{CPU{N}}) where {N}
    #* Dont implement for this project
    error("Not Implemented Yet")
end


function run(spme::SPME{SingleGPU})
    error("Not Implemented Yet")
end

function run(spme::SPME{MultiGPU{N}}) where N
    error("Not Implemented Yet")
end