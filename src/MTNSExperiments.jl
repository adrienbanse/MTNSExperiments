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

ε = 0.5
learning_rate = 0.01
n_episodes_conv = 4000
n_episodes_short = n_episodes_conv # maybe change
max_episode_length = 100
eval_every = 100
n_eval_traj = 10000

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
    verbose = true
)
policy_target, rewards_target = solve(solver_target, target)
println("(REF) Target learned")
# reward_target = evaluate(target, solver_target, policy_target)
println("(REF) Initial reward = $(rewards_target[1])")
initial = rewards_target[1]

n_sources = 100
couples = []
for k = 1:n_sources
    tprob = rand()
    low = tprob < 0.5 ? true : false

    source = SimpleGridWorld(tprob = tprob, rewards = Dict(GWPos(4, 4) => 10.))

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

    # reward on target?
    solver_target_TL = deepcopy(solver_target)
    solver_target_TL.Q_vals = Q_save
    policy_target_TL, rewards_target_TL = solve(solver_target_TL, target)
    println("($k/$n_sources) Target learned with TL")
    println("($k/$n_sources) Initial reward = $(rewards_target_TL[1])")

    jumpstart = rewards_target_TL[1] - initial
    push!(couples, (d, jumpstart, low))
    println("($k/$n_sources) !Jumpstart! = $(jumpstart)")
end

using Plots
p = scatter()
xlabel!("\$\\mathbf{d}(M_{T}, M_{S, i})\$")
ylabel!("Jumpstart reward")
c_green = [(d, j) for (d, j, l) ∈ couples if l]
c_red = [(d, j) for (d, j, l) ∈ couples if !l]

scatter!(p, c_green, color = "green", label = "\$\\delta < 1/2\$")
scatter!(p, c_red, color = "red", label = "\$\\delta \\geq 1/2\$")
display(p)

end # module MTNSExperiments
