function particle_particle(spme::SPME{SingleThread})
   error("Not Implemented Yet")
end

function particle_particle(spme::SPME{CPU{N}}) where {N}
   #* Don't implement for this project
   error("Not Implemented Yet")
end

function particle_particle(spme::SPME{SingleGPU})
   error("Not Implemented Yet")
end

function particle_particle(spme::SPME{MultiGPU{N}}) where {N}
   error("Not Implemented Yet")
end