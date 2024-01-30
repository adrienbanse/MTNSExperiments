function build_action_list!(total::Vector{Vector{P}}, N::Int, save::Vector{P}, A::Vector{P}) where {P}
    if N == 0
        push!(total, save)
        return
    end
    for a ∈ A
        save_cp = push!(copy(save), a)
        build_action_list!(total, N - 1, save_cp, A)
    end
end

function cantor_kantorovich(m1::M, m2::M; N = 3) where {T, P, M <: MDP{T, P}}
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
        w::Vector{Int}, 
        a_list::Vector{P}
    )
        if k == N
            return 2 * min(p1, p2)
        end
        sum = 0.
        for s ∈ S
            p1_new = p1 * T1[a_list[k]][w[end], stateindex(m1, s)]
            p2_new = p2 * T2[a_list[k]][w[end], stateindex(m2, s)] # m1 or m2 shouldn't change anything here    
            r_new = min(p1_new, p2_new)
            if r_new != 0
                sum += ck_rec(k + 1, p1_new, p2_new, r_new, push!(copy(w), stateindex(m1, s)), a_list)
            end
        end
        return r + 0.5 * sum
    end

    a_list_list = Vector{Vector{P}}()
    build_action_list!(a_list_list, N - 1, Vector{P}(), [a for a ∈ A])
    
    best_cand = 0.
    for a_list ∈ a_list_list
        S_rest = 0.
        for s ∈ S
            μ1 = pdf(initialstate(m1), s)
            μ2 = pdf(initialstate(m2), s)  
            r = min(μ1, μ2)
            if r != 0
                S_rest += ck_rec(1, μ1, μ2, r, [stateindex(m1, s)], a_list)
            end
        end
        cand = 0.5 - 0.25 * S_rest
        best_cand = max(cand, best_cand)
    end
    
    return best_cand
end


