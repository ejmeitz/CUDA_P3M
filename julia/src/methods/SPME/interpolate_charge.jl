function interpolate_charge(spme::SPME{SingleThread})

end

function interpolate_charge(spme::SPME{CPU{N}}) where {N}

end

function interpolate_charge(spme::SPME{SingleGPU})

end

function interpolate_charge(spme::SPME{MultiGPU{N}}) where {N}

end