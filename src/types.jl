
struct Atom{D,P,M,C}
    position::SVector{D,P}
    mass::M
    charge::C
end

function Atom(position, mass, charge = 0.0u"q")
    return Atom{length(position), eltype(position), typeof(mass), typeof(charge)}(
        position, mass, charge
    )
end


struct P3M
    pp::PP
    pm::PM
end

struct PM
    #params
end

struct PP
    #params
end