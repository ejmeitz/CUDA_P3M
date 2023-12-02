function run(spme::SPME{SingleThread})

    E_dir, F_dir = particle_particle(spme)
    E_rec, F_rec = particle_mesh(spme)
    E_self = self_energy(spme)

    E_SPME = E_dir + E_rec + E_self
    F_SPME = F_dir .+ F_rec

    return E_SPME, F_SPME

end

function run(spme::SPME{CPU{N}}) where {N}
    #* Dont implement for this project
end


function run(spme::SPME{SingleGPU})

end

function run(spme::SPME{MultiGPU})

end