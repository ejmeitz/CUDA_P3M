
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

#Overload call operator to evalulate energies/forces
(p3m::P3M)(atoms::StructArray{Atom}) = eval_P3M(p3m.pp, p3m.pm, atoms)


struct PM
    #params
end

struct PP
    #params
end