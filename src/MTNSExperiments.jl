module MTNSExperiments

using POMDPs
using TabularTDLearning
using POMDPModels
using POMDPTools

using Random
using Distributions
using RandomMatrix

rng = Random.GLOBAL_RNG

using StatsBase

Random.seed!(123)

include("cantor_kantorovich.jl")
include("q_learning.jl")

function evaluate(mdp, solver, policy)
    # rng = Random.default_rng()
    sim = RolloutSimulator(max_steps = solver.max_episode_length)
    r_tot = 0.0
    for _ ∈ 1:solver.n_eval_traj
        r_tot += simulate(sim, mdp, policy)
    end
    return r_tot / solver.n_eval_traj
end

ε = 0.1
learning_rate = 0.01
n_episodes_conv = 100000
n_episodes_short = 10000
max_episode_length = 100 
eval_every = 100
n_eval_traj = 1000

target = SimpleGridWorld(rewards = Dict(GWPos(4, 4) => 10.), tprob = 0.5)
# solve target
exppolicy_target = EpsGreedyPolicy(target, ε)
solver_target = QLearningSolver(
    exploration_policy = exppolicy_target, 
    learning_rate = learning_rate, 
    n_episodes = n_episodes_short, 
    max_episode_length = max_episode_length, 
    eval_every = eval_every, 
    n_eval_traj = n_eval_traj,
    verbose = false
)
policy_target, _ = solve(solver_target, target)
println("(REF) Target learned")
reward_target = evaluate(target, solver_target, policy_target)
println("(REF) Reward = $reward_target")

n_sources = 100
couples = []
for k = 1:n_sources
    goal = GWPos(rand(1:target.size[1]), rand(1:target.size[2]))
    tprob = rand()
    source = SimpleGridWorld(rewards = Dict(goal => 10.), tprob = 0.5)

    # solve source
    exppolicy_source = EpsGreedyPolicy(source, ε)
    solver_source = QLearningSolver(
        exploration_policy = exppolicy_source, 
        learning_rate = learning_rate, 
        n_episodes = n_episodes_conv, 
        max_episode_length = max_episode_length, 
        eval_every = eval_every, 
        n_eval_traj = n_eval_traj,
        verbose = false
    )
    policy_source, _ = solve(solver_source, source)
    Q_save = policy_source.value_table
    println("($k/$n_sources) Source learned")

    # compute distance
    π_source = Dict{Int, Symbol}()
    for s ∈ states(source)
        π_source[stateindex(source, s)] = action(policy_source, s)
    end
    d = cantor_kantorovich(source, target, π_source; N = 8)
    println("($k/$n_sources) Distance computed d = $d")
    
    # # solve target TL
    # solver_target_TL = deepcopy(solver_target)
    # solver_target_TL.Q_vals = Q_save
    # policy_target_TL, _ = solve(solver_target_TL, target)
    # println("($k/$n_sources) Target learned with TL")
    # reward_target_TL = evaluate(target, solver_target_TL, policy_target_TL)
    # println("($k/$n_sources) Reward = $reward_target_TL")
    # println("($k/$n_sources) Δ = $(reward_target_TL - reward_target)")

    # reward on target?
    policy_target_TL, _ = solve(solver_target, target; past_policy = policy_source)
    println("($k/$n_sources) Target learned with TL")
    reward_target_TL = evaluate(target, solver_source, policy_source)
    println("($k/$n_sources) Reward = $reward_target_TL")
    push!(couples, (d, reward_target_TL))
end

using Plots
p = scatter(legend = false)
for c ∈ couples 
    scatter!(p, c)
end
display(p)

end # module MTNSExperiments
