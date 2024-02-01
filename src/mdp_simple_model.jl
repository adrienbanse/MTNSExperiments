mutable struct SimpleMDP <: MDP{Int64, Int64}
    T::Array{Float64,3}     # SPxAxS    transition matrices
    μ::Vector{Float64}      # S         initial measure vector
    R::Matrix{Float64}      # SxA       rewards
    discount::Float64       # /         discount factor
end

struct DiscreteDistribution{P<:AbstractVector{Float64}}
    p::P
end

POMDPs.support(d::DiscreteDistribution) = 1:length(d.p)
POMDPs.pdf(d::DiscreteDistribution, sp::Int64) = d.p[sp] # T(s', a, s)
Base.rand(rng::AbstractRNG, d::DiscreteDistribution) = sample(rng, Weights(d.p))
POMDPs.states(p::SimpleMDP) = 1:size(p.T, 1)
POMDPs.actions(p::SimpleMDP) = 1:size(p.T, 2)
POMDPs.stateindex(::SimpleMDP, s::Int64) = s
POMDPs.actionindex(::SimpleMDP, a::Int64) = a
POMDPs.discount(p::SimpleMDP) = p.discount
POMDPs.transition(p::SimpleMDP, s::Int64, a::Int64) = DiscreteDistribution(view(p.T, :, a, s))
POMDPs.reward(prob::SimpleMDP, s::Int64, a::Int64) = prob.R[s, a]
POMDPs.initialstate(p::SimpleMDP) = DiscreteDistribution(p.μ)

