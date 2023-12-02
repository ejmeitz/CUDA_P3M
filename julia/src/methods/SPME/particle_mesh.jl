function particle_mesh(spme::SPME{SingleThread})

end

function particle_mesh(spme::SPME{CPU{N}}) where {N}
    #* Don't implement for this project
end

function particle_mesh(spme::SPME{SingleGPU})
    
end

function particle_mesh(spme::SPME{MultiGPU{N}}) where {N}
    
end