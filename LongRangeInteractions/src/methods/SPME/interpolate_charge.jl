function interpolate_charge!(Q, dQdr, spme::SPME{SingleThread})

end

function interpolate_charge!(Q, dQdr, spme::SPME{CPU{N}}) where {N}

end



function charge_kernel(q, r)

end

function interpolate_charge!(Q, dQdr, spme::SPME{SingleGPU})

end

function interpolate_charge!(Q, dQdr, spme::SPME{MultiGPU{N}}) where {N}

end