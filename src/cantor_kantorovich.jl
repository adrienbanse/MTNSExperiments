function cantor_kantorovich(m1::M, m2::M; N = 3) where {T, P, M <: MDP{T, P}}
    S = MTNSExperiments.states(m1)
    A = actions(m1)
    if !issetequal(S, MTNSExperiments.states(m2)) || !issetequal(A, actions(m2))
        throw(AssertionError)
    end
    T1 = transition_matrices(m1)
    T2 = transition_matrices(m2)

    R1 = m1.rewards
    R2 = m2.rewards

    function ck_rec(k::Int, m::Float64, w::Vector{Int}, p1::Float64, p2::Float64, reward::Float64)
        w_new_list = Vector{Vector{Int}}()
        for s in S
            w_new = push!(copy(w), stateindex(m1, s))
            push!(w_new_list, w_new)
        end
        r = Vector{Float64}()
        p1_list = Vector{Float64}()
        p2_list = Vector{Float64}()        

        a = A[1]

        for s in S
            if k == 0
                p1_new = pdf(initialstate(m1), s)
                p2_new = pdf(initialstate(m1), s)
            else
                p1_new = p1 * T1[a][w[end], stateindex(m1, s)]
                p2_new = p2 * T2[a][w[end], stateindex(m1, s)]
            end
            push!(p1_list, p1_new)
            push!(p2_list, p2_new)
            push!(r, min(p1_new, p2_new))

            rew1 = haskey(R1, s) ? R1[s] : 0
            rew2 = haskey(R2, s) ? R2[s] : 0
            reward += (0.5) ^ k * p1_new * p2_new * abs(rew1 - rew2)
        end     

        res = (2.0)^(-(k + 1)) * (m - sum(r))
        if k + 1 == N
            return res, reward
        end
        
        for s ∈ S
            idx = stateindex(m1, s)
            if r[idx] != 0
                inc, _ = ck_rec(k + 1, r[idx], w_new_list[idx], p1_list[idx], p2_list[idx], reward)
                res += inc
            end
        end

        return res, reward
    end

    # try_list = Vector{Vector{P}}()
    # build_action_list!(try_list, N - 1, Vector{P}(), [a for a ∈ A])

    # best_cand = -1
    # best_rew = -1

    # for a_list ∈ try_list
    #     cand, rew = ck_rec(0, 1.0, Vector{Int}(), 1., 1., 0.)
    #     if cand > best_cand
    #         best_cand = cand
    #         best_rew = rew
    #     end
    # end

    res, rew = ck_rec(0, 1.0, Vector{Int}(), 1., 1., 0.)

    α = 0.5
    β = 0.5

    return α * best_cand + β * best_rew
end

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


