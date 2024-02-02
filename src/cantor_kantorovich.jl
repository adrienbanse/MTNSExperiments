function cantor_kantorovich(m1::M, m2::M, policy::Dict{Int, P}; N = 3) where {T, P, M <: MDP{T, P}}
    S = MTNSExperiments.states(m1)
    A = actions(m1)
    if !issetequal(S, MTNSExperiments.states(m2)) || !issetequal(A, actions(m2))
        throw(AssertionError)
    end
    T1 = transition_matrices(m1)
    T2 = transition_matrices(m2)

    function ck_rec(
        k::Int,
        p1::Float64, 
        p2::Float64,
        r::Float64,
        w::Vector{Int}
    )
        if k == N
            return 2 * min(p1, p2)
        end
        sum = 0.
        for s ∈ S
            a = policy[w[end]]
            p1_new = p1 * T1[a][w[end], stateindex(m1, s)]
            p2_new = p2 * T2[a][w[end], stateindex(m2, s)] # m1 or m2 shouldn't change anything here    
            r_new = min(p1_new, p2_new)
            if r_new != 0
                sum += ck_rec(k + 1, p1_new, p2_new, r_new, push!(copy(w), stateindex(m1, s)))
            end
        end
        return r + 0.5 * sum
    end

    S_rest = 0.
    for s ∈ S
        μ1 = pdf(initialstate(m1), s)
        μ2 = pdf(initialstate(m2), s)  
        r = min(μ1, μ2)
        if r != 0
            S_rest += ck_rec(1, μ1, μ2, r, [stateindex(m1, s)])
        end
    end

    return 0.5 - 0.25 * S_rest
end


