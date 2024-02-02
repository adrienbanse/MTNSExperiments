function TabularTDLearning.solve(solver::QLearningSolver, mdp::MDP)
    (;rng,exploration_policy) = solver
    γ = discount(mdp)
    Q = if isnothing(solver.Q_vals)
        zeros(length(states(mdp)), length(actions(mdp)))
    else
        solver.Q_vals
    end::Matrix{Float64}

    sim = RolloutSimulator(rng=rng, max_steps=solver.max_episode_length)

    on_policy = ValuePolicy(mdp, Q)
    k = 0

    rewards = []
    for i = 1:solver.n_episodes
        s = rand(rng, initialstate(mdp))
        t = 0
        while !isterminal(mdp, s) && t < solver.max_episode_length
            a = action(exploration_policy, on_policy, k, s)
            k += 1
            sp, r = @gen(:sp, :r)(mdp, s, a, rng)
            si = stateindex(mdp, s)
            ai = actionindex(mdp, a)
            spi = stateindex(mdp, sp)
            Q[si, ai] += solver.learning_rate * (r + γ * maximum(@view(Q[spi, :])) - Q[si,ai])
            s = sp
            t += 1
        end
        if i % solver.eval_every == 0
            r_tot = 0.0
            for _ in 1:solver.n_eval_traj
                r_tot += simulate(sim, mdp, on_policy, rand(rng, initialstate(mdp)))
            end
            push!(rewards, r_tot / solver.n_eval_traj)
            solver.verbose && println("On Iteration $i, Returns: $(r_tot / solver.n_eval_traj)")
        end
    end
    return on_policy, rewards
end
