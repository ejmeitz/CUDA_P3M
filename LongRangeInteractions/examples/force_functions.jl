

function LJ(r, ϵ::Float32, σ::Float32)
    k = (σ/r)^6
    U = 4*ϵ*(k*(k-1))
    F = -4*ϵ*(12*(k/r) + 6*(k/r))
    return U, F
end