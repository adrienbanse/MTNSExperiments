module MTNSExperiments

using POMDPs
using TabularTDLearning
using POMDPModels
using POMDPTools

using Random
using Distributions
using RandomMatrix

using StatsBase

Random.seed!(12)
include("cantor_kantorovich.jl")
include("q_learning.jl")

struct TabularExtension <: MDP{Int, Int}
    mdp::TabularMDP
    μ::Vector{Float64}
end
struct DiscreteDistribution{P<:AbstractVector{Float64}}
    p::P
end
Base.rand(rng::AbstractRNG, d::DiscreteDistribution) = sample(rng, Weights(d.p))
POMDPs.initialstate(p::TabularExtension) = DiscreteDistribution(p.μ)
POMDPs.support(d::DiscreteDistribution) = 1:length(d.p)
POMDPs.pdf(d::DiscreteDistribution, sp::Int64) = d.p[sp] # T(s', a, s)
POMDPs.states(p::TabularExtension) = 1:size(p.mdp.T, 1)
POMDPs.actions(p::TabularExtension) = 1:size(p.mdp.T, 2)
POMDPs.stateindex(::TabularExtension, s::Int64) = s
POMDPs.actionindex(::TabularExtension, a::Int64) = a
POMDPs.discount(p::TabularExtension) = p.mdp.discount
POMDPs.transition(p::TabularExtension, s::Int64, a::Int64) = DiscreteDistribution(view(p.mdp.T, :, a, s))
POMDPs.reward(prob::TabularExtension, s::Int64, a::Int64) = prob.mdp.R[s, a]

function MyRandomMDP(ns::Int64, na::Int64, R::Matrix{Float64}, discount::Float64; rng::AbstractRNG = Random.GLOBAL_RNG)
    T = zeros(ns, na, ns)
    for a ∈ 1:na
        T[:, a, :] = randStochastic(ns; type = 1)
    end
    μ = rand(rng, ns)
    μ ./= sum(μ)
    mdp = TabularMDP(T, R, discount)
    return TabularExtension(mdp, μ)
end




# include("q_learning.jl")

# tprob_target = rand()
# mdp_target = SimpleGridWorld(tprob = tprob_target)
# tprob_source = rand()
# mdp_source = SimpleGridWorld(tprob = tprob_source)

# rng = MersenneTwister(1)
# ε = 0.5
# learning_rate = 0.05
# n_episodes = 200
# max_episode_length = 50
# eval_every = 1
# n_eval_traj = 5000

# using Plots
# p = plot()
# for i = 1:2
#     global p

#     exppolicy_source = EpsGreedyPolicy(mdp_source, ε, rng = rng)
#     solver_source = QLearningSolver(
#         exploration_policy = exppolicy_source, 
#         learning_rate = learning_rate, 
#         n_episodes = n_episodes, 
#         max_episode_length = max_episode_length, 
#         eval_every = eval_every, 
#         n_eval_traj = n_eval_traj,
#         rng = rng,
#         verbose = false
#     )
#     policy_source, rewards_source = solve(solver_source, mdp_source)
#     Q_save = policy_source.value_table


#     # second, solve target without anything
#     exppolicy_target_without = EpsGreedyPolicy(mdp_target, ε)
#     solver_target_without = QLearningSolver(
#         exploration_policy = exppolicy_target_without, 
#         learning_rate = learning_rate, 
#         n_episodes = n_episodes, 
#         max_episode_length = max_episode_length, 
#         eval_every = eval_every, 
#         n_eval_traj = n_eval_traj, 
#         rng = rng,
#         verbose = false
#     )
#     policy_target_without, rewards_without = solve(solver_target_without, mdp_target)


#     # third, solve target with inital source Q
#     solver_target_with = deepcopy(solver_target_without)
#     solver_target_with.Q_vals = Q_save
#     policy_target_with, rewards_with = solve(solver_target_with, mdp_target)

#     # plot!(p, rewards_source, label = "source")
#     plot!(p, rewards_without, label = "without")
#     plot!(p, rewards_with, label = "with")
# end
# display(p)


couples = []
n_exp = 20

size = (8, 8)
# rewards = Dict(GWPos(4,3)=>-10.0, GWPos(4,6)=>-5.0, GWPos(2,3)=>10.0, GWPos(7,7)=>3.0)

rewards_target = Dict(GWPos(rand(1:size[1]), rand(1:size[1])) => 10.)
# tprob_target = rand()
mdp_target = SimpleGridWorld(size = size, rewards = rewards_target, tprob = 0.5, discount = 0.5)

d = 0.
println("0 / $n_exp: CK-distance computed")

r = q_learning_experiments(mdp_target, mdp_target)
println("0 / $n_exp: advantage computed")

push!(couples, (d, r))
println("0 / $n_exp: done (d = $d, Δr = $r)\n")

for k = 1:n_exp
    global couples

    rewards_source = Dict(GWPos(rand(1:size[1]), rand(1:size[1])) => 10.)
    tprob_source = rand()
    mdp_source = SimpleGridWorld(size = size, rewards = rewards_source, tprob = tprob_source, discount = 0.5)

    d = cantor_kantorovich(mdp_source, mdp_target; N = 5)
    println("$k / $n_exp: CK-distance computed")

    r = q_learning_experiments(mdp_source, mdp_target)
    println("$k / $n_exp: advantage computed")

    push!(couples, (d, r))
    println("$k / $n_exp: done (d = $d, Δr = $r)\n")
end

using Plots
p = scatter(legend = false)
for c ∈ couples
    scatter!(p, c)
end
display(p)

end # module MTNSExperiments
