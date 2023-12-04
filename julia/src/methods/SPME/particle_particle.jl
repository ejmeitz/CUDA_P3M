function particle_mesh(spme::SPME{SingleThread})

end

function particle_particle(spme::SPME{CPU{N}}) where {N}
   #* Don't implement for this project
end

function particle_particle(spme::SPME{SingleGPU})
    
end

function particle_particle(spme::SPME{MultiGPU{N}}) where {N}
    
end