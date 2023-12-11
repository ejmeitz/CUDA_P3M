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