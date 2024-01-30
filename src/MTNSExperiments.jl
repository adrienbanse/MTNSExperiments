module MTNSExperiments

using POMDPs
using TabularTDLearning
using POMDPModels
using POMDPTools
using Random
using Distributions

Random.seed!(1234)
include("cantor_kantorovich.jl")

function make_stochastic!(P::Matrix{Float64})
    s, _ = size(P)
    for i = 1:s
        P[i, :] /= sum(P[i, :])
    end
end

Base.@kwdef struct SimpleMDP <: MDP{Int, Symbol} 
    state_size::Int
    P::Dict{Symbol, Matrix{Float64}}
    rewards::Dict{Int, Float64}
    discount::Float64
end

POMDPs.states(mdp::SimpleMDP) = collect(1:mdp.state_size)
POMDPs.stateindex(mdp::SimpleMDP, s::Int) = s
struct SimpleUniform 
    state_size::Int
end
Base.rand(rng::AbstractRNG, d::SimpleUniform)::Int = rand(rng, 1:d.state_size)
POMDPs.pdf(d::SimpleUniform, s::Int) = 1 / d.state_size
POMDPs.support(d::SimpleUniform) = (x for x ∈ 1:d.state_size)
POMDPs.initialstate(mdp::SimpleMDP) = SimpleUniform(mdp.state_size)
POMDPs.actions(mdp::SimpleMDP) = (:a, :b)
Base.rand(rng::AbstractRNG, t::NTuple{L,Symbol}) where L = t[rand(rng, 1:length(t))]
POMDPs.actionindex(mdp::SimpleMDP, a::Symbol) = a == :a ? 1 : 2
POMDPs.isterminal(m::SimpleMDP, s::Int) = false
POMDPs.transition(mdp::SimpleMDP, s::Int, a::Symbol) = (mdp.P)[a]
transition_matrices(mdp::SimpleMDP) = mdp.P
POMDPs.reward(mdp::SimpleMDP, s::Int) = get(mdp.rewards, s, 0.0)
POMDPs.reward(mdp::SimpleMDP, s::Int, a::Symbol) = reward(mdp, s)
POMDPs.discount(mdp::SimpleMDP) = mdp.discount
function POMDPs.convert_a(::Type{V}, a::Symbol, m::SimpleMDP) where {V<:AbstractArray}
    convert(V, [actionindex(m, a)])
end
function POMDPs.convert_a(::Type{Symbol}, vec::V, m::SimpleMDP) where {V<:AbstractArray}
    actions(m)[convert(Int, first(vec))]
end

function generate_random_MDP(
    state_size::Int, 
    γ::Float64, 
    r_range::Tuple{Float64, Float64}
)
    rewards = Dict{Int, Float64}() 
    for i = 1:state_size
        rewards[i] = rand(Distributions.Uniform(r_range...))
    end
    P = Dict{Symbol, Matrix{Float64}}()
    P[:a] = rand(Distributions.Uniform(0. ,1.), state_size, state_size)
    make_stochastic!(P[:a])
    P[:b] = rand(Distributions.Uniform(0. ,1.), state_size, state_size)
    make_stochastic!(P[:b])
    return SimpleMDP(state_size, P, rewards, γ)
end

function evaluate(mdp, solver, policy)
    # rng = Random.default_rng()
    sim = RolloutSimulator(max_steps = solver.max_episode_length)
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

mdp_target = generate_random_MDP(10, 0.8, (3., 5.))

couples = []
n_sources = 10
for k = 1:n_sources
    global couples, t_prob

    mdp_source = mdp_target = generate_random_MDP(10, 0.8, (3., 5.))
    
    # goal_source = (rand(1:world_size), rand(1:world_size))
    # tprob_source = rand(Distributions.Uniform(0, 1))
    # mdp_source = SimpleGridWorld(size = (world_size, world_size), rewards = Dict(GWPos(goal_source...) => 1.0), tprob = tprob_source) 

    d = @time cantor_kantorovich(mdp_source, mdp_target; N = 5)
    println("$k / $n_sources: CK-distance computed")
    r = q_learning_experiments(mdp_source, mdp_target; n_experiments = 1)
    println("$k / $n_sources: advantage computed")
    push!(couples, (d, r))
    println("$k / $n_sources: done (d = $d, Δr = $r)")
    println()
end

using Plots
p = scatter(legend = false)
for c ∈ couples
    scatter!(p, c)
end
display(p)

end # module MTNSExperiments
