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

function b(mⱼ,n,Kⱼ)
    
end

#equivalent, time to see whats faster
# def M(u, n):
#     return (1/math.factorial(n-1)) * np.sum([((-1)**k)*math.comb(n,k)*np.power(max(u-k, 0), n-1) for k in range(n+1)])
