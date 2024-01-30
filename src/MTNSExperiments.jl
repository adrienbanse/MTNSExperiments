module MTNSExperiments

using POMDPs
using TabularTDLearning
using POMDPModels
using POMDPTools
using Random
using Distributions

Random.seed!(1234)
include("cantor_kantorovich.jl")

function evaluate(mdp, solver, policy)
    rng = Random.default_rng()
    sim = RolloutSimulator(rng=rng, max_steps = solver.max_episode_length)
    r_tot = 0.0
    for _ ∈ 1:solver.n_eval_traj
        r_tot += simulate(sim, mdp, policy)
    end
    return r_tot / solver.n_eval_traj
end

# use Q-Learning
function q_learning_experiments(
    mdp_source::MDP, 
    mdp_target::MDP; 
    ε = 0.05, 
    learning_rate = 0.1, 
    n_episodes = 10000, 
    max_episode_length = 50, 
    eval_every = 50, 
    n_eval_traj = 100, 
    n_experiments = 10
)
    mean = 0
    for _ = 1:n_experiments
        # first solve source
        exppolicy_source = EpsGreedyPolicy(mdp_source, ε)
        solver_source = QLearningSolver(
            exploration_policy = exppolicy_source, 
            learning_rate = learning_rate, 
            n_episodes = n_episodes, 
            max_episode_length = max_episode_length, 
            eval_every = eval_every, 
            n_eval_traj = n_eval_traj,
            verbose = false
        )
        policy_source = solve(solver_source, mdp_source)
        Q_save = policy_source.value_table

        # second, solve target without anything
        exppolicy_target_without = EpsGreedyPolicy(mdp_target, ε)
        solver_target_without = QLearningSolver(
            exploration_policy = exppolicy_target_without, 
            learning_rate = learning_rate, 
            n_episodes = n_episodes, 
            max_episode_length = max_episode_length, 
            eval_every = eval_every, 
            n_eval_traj = n_eval_traj, 
            verbose = false
        )
        policy_target_without = solve(solver_target_without, mdp_target)

        # third, solve target with inital source Q
        solver_target_with = deepcopy(solver_target_without)
        solver_target_with.Q_vals = Q_save
        policy_target_with = solve(solver_target_with, mdp_target)

        mean += evaluate(mdp_target, solver_target_with, policy_target_with) - evaluate(mdp_target, solver_target_without, policy_target_without)
    end
    return mean / n_experiments
end

world_size = 20

goal_target = (rand(1:world_size), rand(1:world_size))
tprob_target = rand(Distributions.Uniform(0, 1))
mdp_target = SimpleGridWorld(size = (world_size, world_size), rewards = Dict(GWPos(goal_target...) => 1.0), tprob = tprob_target)

couples = []
n_sources = 20
for k = 1:n_sources
    global couples
    
    goal_source = (rand(1:world_size), rand(1:world_size))
    tprob_source = rand(Distributions.Uniform(0, 1))
    mdp_source = SimpleGridWorld(size = (world_size, world_size), rewards = Dict(GWPos(goal_source...) => 1.0), tprob = tprob_source) 

    d = cantor_kantorovich(mdp_source, mdp_target; N = 4)
    println("$k / $n_sources: CK-distance computed")
    r = q_learning_experiments(mdp_source, mdp_target)
    println("$k / $n_sources: advantage computed")
    push!(couples, (d, r))
    println("$k / $n_sources: done (d = $d, Δr = $r)")
    println()
end

using Plots
p = scatter()
for c ∈ couples
    scatter!(p, c)
end
display(p)

end # module MTNSExperiments
