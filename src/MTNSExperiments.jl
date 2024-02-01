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

Random.seed!(12)

include("cantor_kantorovich.jl")
include("mdp_simple_model.jl")

ns = 4
na = 2

T = zeros(ns, na, ns)
for a ∈ 1:na
    T[:, a, :] = 1 / ns * ones(ns, ns)
end
R = zeros(ns, na)
R[1, :] .= 1.; R[4, :] .= -1.
μ = 1 / ns * ones(ns)
discount = 0.5
ref = SimpleMDP(T, μ, R, discount)

# # Modify the probability Matrix
# Tbis = copy(T)
# Tbis[1, 1, :] = [
#     0.2 0.4 0.4 0.2
# ]
# # for i = 1:ns
# #     Tbis[:,1,i] /= sum(Tbis[:,1,i])
# # end
# m1 = SimpleMDP(Tbis, μ, R, discount)
# @time d1 = cantor_kantorovich(ref, m1; N = 9)
# println(d1)

# # Modify the initial vector 
# μbis = [0.2, 0.4, 0.4, 0.2]
# m2 = SimpleMDP(T, μbis, R, discount)
# @time d2 = cantor_kantorovich(ref, m2; N = 9)
# println(d2)

# Modify the rewards 1
Rbis1 = R * 10.
m3 = SimpleMDP(T, μ, Rbis1, discount)
@time d3 = cantor_kantorovich(ref, m3; N = 9)
println(d3)

# # Modify the rewards 2
# Rbis2 = zeros(ns, na)
# Rbis2[2, :] .= 1.; Rbis2[3, :] .= -1.
# m4 = SimpleMDP(T, μ, Rbis2, discount)
# @time d4 = cantor_kantorovich(ref, m4; N = 9)
# println(d4)

end # module MTNSExperiments
