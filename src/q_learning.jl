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

            push!(rewards, r_tot/solver.n_eval_traj)

            solver.verbose && println("On Iteration $i, Returns: $(r_tot/solver.n_eval_traj)")
        end
    end
    return on_policy, rewards
end

function q_learning_experiments(
    mdp_source::MDP, 
    mdp_target::MDP; 
    ε = 0, 
    learning_rate = 0.01, 
    n_episodes_conv = 1000000, 
    n_episodes_stop = 1000,
    max_episode_length = 100, 
    eval_every = 100, 
    n_eval_traj = 100
)
    # first solve source
    exppolicy_source = EpsGreedyPolicy(mdp_source, ε)
    solver_source = QLearningSolver(
        exploration_policy = exppolicy_source, 
        learning_rate = learning_rate, 
        n_episodes = n_episodes_conv, 
        max_episode_length = max_episode_length, 
        eval_every = eval_every, 
        n_eval_traj = n_eval_traj,
        verbose = false
    )
    policy_source, _ = solve(solver_source, mdp_source)
    Q_save = policy_source.value_table

    # second, solve target without anything
    exppolicy_target_without = EpsGreedyPolicy(mdp_target, ε)
    solver_target_without = QLearningSolver(
        exploration_policy = exppolicy_target_without, 
        learning_rate = learning_rate, 
        n_episodes = n_episodes_stop, 
        max_episode_length = max_episode_length, 
        eval_every = eval_every, 
        n_eval_traj = n_eval_traj, 
        verbose = false
    )
    policy_without, rewards_without = solve(solver_target_without, mdp_target)

    # third, solve target with inital source Q
    solver_target_with = deepcopy(solver_target_without)
    solver_target_with.Q_vals = Q_save
    policy_with, rewards_with = solve(solver_target_with, mdp_target)

    # fourth, compute the ground truth Q for target
    solver_target_gt = QLearningSolver(
        exploration_policy = exppolicy_target_without, 
        learning_rate = learning_rate, 
        n_episodes = n_episodes_conv, 
        max_episode_length = max_episode_length, 
        eval_every = eval_every, 
        n_eval_traj = n_eval_traj, 
        verbose = false
    )
    policy_gt, rewards_gt = solve(solver_target_gt, mdp_target)

    # compute distances to gt
    d_without_to_gt = sum((policy_without.value_table .- policy_gt.value_table).^2) / length(Q_save)
    d_with_to_gt = sum((policy_with.value_table .- policy_gt.value_table).^2) / length(Q_save)

    # It should be that  d_with_to_gt < d_without_to_gt, so let's use d_without_to_gt - d_with_to_gt to compute performance

    return d_without_to_gt - d_with_to_gt
end

