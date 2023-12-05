function M(u, n)
    if n > 2
        return (u/(n-1))*M(u,n-1) + ((n-u)/(n-1))*M(u-1,n-1)
    else if n == 2
        if u >= 0 && u <= 2
            return 1 - abs(u-1)
        else
            return 0
        end
    else
        println("Shouldn't be here")
    end
end

function dMdu(u,n)
    return M(u, n-1) - M(u-1, n-1)
end

#Remove allocation? probably not a huge deal this is precalculated
function b(mⱼ,n,Kⱼ)
    m_K = mⱼ/Kⱼ
    num = exp(2*π*1im*(n-1)*m_K)
    denom = sum([M(k+1, n)*exp(2*π*1im*k*m_K) for k in range(0,n-2)])
    return  num/denom 
end

function calc_C(β, V, ms, recip_lat)
    m_star = ms[1]*recip_lat[1,:] + ms[2]*recip_lat[2,:] + ms[3]*recip_lat[3,:]
    m_sq = dot(m_star,m_star)
    return (1/(π*V))*(exp(-(π^2)*m_sq/(β^2))/m_sq)
end
    
function abs2(x::Complex)
    return (real(x)^2) + (imag(x)^2)
end

function calc_BC(sys::System, spme::SPME)
    V = vol(sys)
    K1, K2, K3 = n_mesh(spme)
    recip_lat = reciprocal_lattice(lattice_vec(sys)) #SPME uses the 1 normalized version
    n = spline_order(spme)

    BC = zeros(Complex64, K1, K2, K3)
    hs = [0.0, 0.0, 0.0]

    for m1 in range(K1)
        hs[1] = (m1 <= (K1/2) ? m1 : m1 - K1)
        B1 = abs2(b(m1,K1,n))
        for m2 in range(K2)
            hs[2] = ((m2 <= (K2/2)) ? m2 : m2 - K2)
            B2 = B1*abs2(b(m2,K2,n))
            for m3 in range(K3)
                hs[3] = ((m3 <= (K3/2)) ? m3 : m3 - K3)

                if m1 == 0 && m2 == 0 && m3 == 0
                    continue
                end
                
                B3 = B2*abs2(b(m3,K3,n))
                C = calc_C(spme.β, V, hs, recip_lat)

                BC[m1,m2,m3] = B3*C
            end
        end
    end

    return BC
end

#equivalent, time to see whats faster
# def M(u, n):
#     return (1/math.factorial(n-1)) * np.sum([((-1)**k)*math.comb(n,k)*np.power(max(u-k, 0), n-1) for k in range(n+1)])
