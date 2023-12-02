function particle_particle(spme::SPME{CPU{N}}) where {N}
    #TODO particle-particle on CPU
end

function particle_particle(spme::SPME{SingleGPU}) where {N}
    #TODO particle-particle on GPU
end

function particle_particle(spme::SPME{MultiGPU}) where {N}
    #TODO particle-particle on GPU
end