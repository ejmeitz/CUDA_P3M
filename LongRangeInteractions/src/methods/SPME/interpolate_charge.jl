export interpolate_charge!

function interpolate_charge!(Q, dQdr, spme::SPME{SingleThread})
    K1,K2,K3 = n_mesh(spme)
    recip_lat = reciprocal_lattice(spme)
    q_arr = charges(spme.sys)
    N_atoms = length(q_arr)
    n = spme.spline_order

    u = scaled_fractional_coords(positions(spme.sys), n_mesh(spme), recip_lat)
    # Q = zeros(K1, K2, K3)
    # dQdr = zeros(N_atoms, 3, K1, K2, K3) #deriv in real space

    for i in 1:N_atoms
        for c0 in 1:n
            l0 = round(Int64,u[i][1]) - c0 # Grid point to interpolate onto

            M0 = M(u[i][1] - l0, n)
            q_n_0 = q_arr[i]*M0 #if 0 <= u_i0 - l0 <= n will be non-zero
            dM0 = dMdu(u[i][1] - l0,n)

            l0 += ceil(Int64,n/2) # Shift
            if l0 < 0 # Apply PBC
                l0 += K1
            elseif l0 >= K1
                l0 -= K1
            end

            for c1 in 1:n
                l1 = round(Int64,u[i][2]) - c1 # Grid point to interpolate onto

                M1 = M(u[i][2] - l1, n)
                q_n_1 = q_n_0*M1 #if 0 <= u_i1 - l1 <= n will be non-zero
                dM1 = dMdu(u[i][2] - l1,n)


                l1 += ceil(Int64,n/2) # Shift
                if l1 < 0 # Apply PBC
                    l1 += K2
                elseif l1 >= K2
                    l1 -= K2
                end
                
                for c2 in range(1,n)
                    l2 = round(Int64,u[i][3]) - c2 # Grid point to interpolate onto

                    M2 = M(u[i][3] - l2, n)
                    q_n_2 = q_n_1*M2 #if 0 <= u_i1 - l1 <= n will be non-zero
                    dM2 = dMdu(u[i][3] - l2,n)

                    l2 += ceil(Int64,n/2) # Shift
                    if l2 < 0 # Apply PBC
                        l2 += K2
                    elseif l2 >= K2
                        l2 -= K2
                    end

                    Q[l0+1,l1+1,l2+1] += q_n_2
                    
                    #*Does it matter that l0,l1,l2 is also a function of r_ia
                    #*This looks like its probably equivalent to some matrix multiply
                    # dQdr[i, 1, l0+1, l1+1, l2+1] = q_arr[i]*(K1*recip_lat[1][1]*dM0*M1*M2 + K2*recip_lat[2][1]*dM1*M0*M2 + K3*recip_lat[3][1]*dM2*M0*M1)
                    # dQdr[i, 2, l0+1, l1+1, l2+1] = q_arr[i]*(K1*recip_lat[1][2]*dM0*M1*M2 + K2*recip_lat[2][2]*dM1*M0*M2 + K3*recip_lat[3][2]*dM2*M0*M1)
                    # dQdr[i, 3, l0+1, l1+1, l2+1] = q_arr[i]*(K1*recip_lat[1][3]*dM0*M1*M2 + K2*recip_lat[2][3]*dM1*M0*M2 + K3*recip_lat[3][3]*dM2*M0*M1)
    
                end
            end
        end
    end
    
    return Q, dQdr
end

                    
# function interpolate_charge!(Q, dQdr, spme::SPME{CPU{N}}) where {N}

# end


# function interpolate_charge!(Q, dQdr, spme::SPME{SingleGPU})

# end

# function interpolate_charge!(Q, dQdr, spme::SPME{MultiGPU{N}}) where {N}

# end