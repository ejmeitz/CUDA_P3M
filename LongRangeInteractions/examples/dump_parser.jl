export LammpsDump, parse_timestep!, parse_next_timestep!, get_col

struct LammpsDump{HD,DD}
    header_length::UInt32
    header_data::HD #TODO: Make NamedTuple
    col_idxs::Dict{String,Integer}
    n_lines::UInt32
    n_samples::UInt32
    data_storage::DD
    path::String
end


function LammpsDump(path; header_length = 9)
    header_data, n_lines = parse_dump_header(path, header_length)
    n_samples = UInt32(n_lines / (header_data["N_atoms"] + header_length))
    # Build dataframe to hold data -- reuse memory
    names = tuple(Symbol.(header_data["fields"])...)
    values = (zeros(header_data["N_atoms"]) for _ in 1:length(header_data["fields"]))
    dump_data = DataFrame( NamedTuple{names}(values));

    #Create dict that maps col names to indices
    col_idxs = Dict(field => i for (i,field) in enumerate(header_data["fields"]))

    return LammpsDump{typeof(header_data), typeof(dump_data)}(header_length, header_data, col_idxs, n_lines, n_samples, dump_data, path)
end

Base.length(ld::LammpsDump) = ld.n_samples
# Base.iterate(ld::LammpsDump, state = 1) = state > length(ld) ? nothing : (parse_timestep!(ld, state), state + 1)
Base.haskey(ld::LammpsDump, x::String) = x ∈ names(ld.data_storage)
get_col(ld::LammpsDump, x::String) = getproperty(ld.data_storage, Symbol(x))

function skiplines(path::String, num_lines)
    io = open(path, "r")
    for _ in range(1, num_lines)
        skipchars(!=('\n'), io)
        read(io, Char)
    end
    return io
end

function readlines_until(path::String, num_lines)
    io = open(path, "r")
    for _ in range(1, num_lines)
        readline(io)
    end
    return io
end

function parse_dump_header(dump_path, dump_header_len)
    file = open(dump_path, "r")
    n_lines = countlines(file) #need to close file after getting this data
    close(file)

    file = open(dump_path, "r")

    #parse first header for useful data
    header_data = Dict("N_atoms" => 0.0, "L_x" => [0.0,0.0], "L_y" => [0.0,0.0],
         "L_z" => [0.0,0.0], "fields" => nothing)
    for i in range(1, dump_header_len)
        line = readline(file)
        if i == 4
            header_data["N_atoms"] = parse(Int, line)
        elseif i == 6
            header_data["L_x"]  = [parse(Float64,x) for x in split(strip(line))]
        elseif i == 7
            header_data["L_y"] = [parse(Float64,y) for y in split(strip(line))]
        elseif i == 8
            header_data["L_z"] = [parse(Float64,z) for z in split(strip(line))]
        elseif i == 9
            header_data["fields"] = split(strip(line))[3:end]
        end
    end

    return header_data, n_lines
end

"""
Opens dump file and pulls out specific sample of data. This is expensive 
if the file is long. 
"""
function parse_timestep!(ld::LammpsDump, sample_number)

    lines_to_skip = (sample_number - 1)*(ld.header_length + ld.header_data["N_atoms"]) + (ld.header_length)
    io = skiplines(ld.path, lines_to_skip)
    #Parse atom data
    for j in range(1, ld.header_data["N_atoms"])
        ld.data_storage[j,:] .= parse.(Float64, split(strip(readline(io))))
    end
    close(io)

    return ld
end

"""
Parse next timestep from `io` and stores it in `ld`. 
"""
function parse_next_timestep!(ld::LammpsDump, io::IOStream)
    
    #Skip header 
    for _ in range(1, ld.header_length) readline(io) end

    #Parse atom data
    for j in range(1, ld.header_data["N_atoms"])
        ld.data_storage[j,:] .= parse.(Float64, split(strip(readline(io))))
    end

    return ld, io

end

"""
Parse `cols` of next timestep from `io` associated with `ld`. Output stored in `out`. 
"""
function parse_next_timestep!(out::Matrix{T}, ld::LammpsDump, io::IOStream, cols::Vector{<:Integer}) where T
    #Skip header 
    for _ in range(1, ld.header_length) readline(io) end

    #Parse atom data
    for j in range(1, ld.header_data["N_atoms"])
        out[j,:] .= parse.(T, split(strip(readline(io)))[cols])
    end

    return out, ld, io
end
