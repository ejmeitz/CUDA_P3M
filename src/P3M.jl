


function eval_P3M(pp::PP, pm::PM, atoms::StructArray{Atom})


end

#Overload call operator to evalulate energies/forces
(p3m::P3M)(atoms::StructArray{Atom}) = eval_P3M(p3m.pp, p3m.pm, atoms)
