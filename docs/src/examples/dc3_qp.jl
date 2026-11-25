# # Training a DC3 proxy on the convex QP of the DC3 paper
#
# This example trains a small neural network with [`DC3`](@ref) on the
# parametric convex QP of Section 4.1 of Donti, Rolnick & Kolter (2021),
# and compares its predictions with the solver optimum. It runs on a CPU in
# well under a minute.
#
# The proxy is a [Lux](https://lux.csail.mit.edu) network; any callable that maps
# a `num_parameters × B` matrix to an `output_dimension × B` matrix and can be
# differentiated with Zygote would do.

using L2ODC3, L2OCore
using JuMP, HiGHS
using LinearAlgebra, Random, Statistics
using Lux, Optimisers, Zygote
import MathOptInterface as MOI

# ## The problem
#
# ```math
# \min_y \tfrac12 y^\top Q y + c^\top y \quad \text{s.t.} \quad A y = x,\quad G y \le h,
# ```
#
# with `x ∈ [-1, 1]^{n_eq}` the parameters. As in the paper, `Q` is diagonal with
# entries in `[0, 1]`, `A` and `G` are standard normal and
# `h = Σⱼ |(G A⁺)ᵢⱼ|`, which makes every instance feasible (eq. (6) of the
# paper). We use `n = 20` variables, 6 equalities and 10 inequalities.

rng = Xoshiro(42)
n, neq, nin = 20, 6, 10
Q = Diagonal(rand(rng, n))
c = rand(rng, n)
A = randn(rng, neq, n)
G = randn(rng, nin, n)
h = vec(sum(abs.(G * pinv(A)); dims=2))

model = Model()
@variable(model, x[1:neq] in MOI.Parameter.(zeros(neq)))
@variable(model, y[1:n])
@constraint(model, A * y .== x)
@constraint(model, G * y .<= h)
@objective(model, Min, 0.5 * y' * Q * y + c' * y)
problem = ParametricProblem(model)

# ## The DC3 pipeline
#
# [`DC3`](@ref) picks `n - n_eq = 14` independent variables (the proxy output),
# completes the other 6 by solving the equalities (one batched matrix product)
# and runs 10 correction steps on the inequality violation during training. At
# test time it runs up to 1000 steps, stopping once every inequality holds to
# `1e-8`.

method = DC3(; λ_ineq=10.0, λ_eq=10.0, train_steps=10, test_steps=1000, test_tol=1e-8)
pipe = L2OPipeline(problem, method)
d = pipe.decomposition
(; independent=independent_indices(d), dependent=dependent_indices(d))

# Training is self-supervised: only parameter samples are needed.

sampler = BoxSampler(-ones(neq), ones(neq))
train = generate_data(pipe; sampler, n=2000, rng)
test = generate_data(pipe; sampler, n=200, rng, optimizer=HiGHS.Optimizer)

# ## The proxy and the training loop

net = Chain(Dense(neq => 64, relu), Dense(64 => 64, relu), Dense(64 => output_dimension(pipe)))
ps, st = Lux.setup(rng, net)
ps = f64(ps)  # the problem data are Float64
proxy(P, ps) = first(net(P, ps, st))

opt = Optimisers.setup(Optimisers.Adam(1e-3), ps)
batch = 100
losses = Float64[]
elapsed = @elapsed for epoch in 1:40
    perm = randperm(rng, num_samples(train))
    for k in 1:batch:num_samples(train)
        Pb = train.inputs[:, perm[k:(k + batch - 1)]]
        ℓ, back = Zygote.pullback(p -> loss(pipe, Pb, proxy(Pb, p)), ps)
        global opt, ps = Optimisers.update(opt, ps, back(one(ℓ))[1])
        push!(losses, ℓ)
    end
end
(; seconds=elapsed, first_loss=losses[1], last_loss=mean(losses[(end - 19):end]))

# ## Evaluation against HiGHS
#
# `complete(pipe, inputs, output)` applies the completion and the test-time
# corrections. The equality constraints hold to round-off by construction
# (`EqualityFeasible`); the inequality violation is driven to zero by the
# correction.

P = test.inputs
pred = complete(pipe, P, proxy(P, ps))
report = evaluate(pipe, P, pred)
gap = (report.objective .- test.objective) ./ abs.(test.objective)
(;
    mean_objective_dc3=mean(report.objective),
    mean_objective_highs=mean(test.objective),
    mean_relative_gap=mean(gap),
    max_equality_violation=maximum(report.max_equality_violation),
    max_inequality_violation=maximum(report.max_inequality_violation),
)

# The method's guarantee is checked numerically:

check(EqualityFeasible(), problem, P, pred; atol=1e-9)

# ## FSNet on the same problem
#
# [`FSNet`](@ref) predicts every variable and runs gradient descent on the total
# violation `‖h‖² + ‖ReLU(g)‖²` (step `1/L`); the loss adds `ρ/2 ‖y - ŷ‖²`.

fpipe = L2OPipeline(problem, FSNet(; ρ=1.0, train_steps=10, test_steps=2000, test_tol=1e-16))
fnet = Chain(Dense(neq => 64, relu), Dense(64 => 64, relu), Dense(64 => n))
fps, fst = Lux.setup(rng, fnet)
fps = f64(fps)
fproxy(P, p) = first(fnet(P, p, fst))
fopt = Optimisers.setup(Optimisers.Adam(1e-3), fps)
for epoch in 1:40
    perm = randperm(rng, num_samples(train))
    for k in 1:batch:num_samples(train)
        Pb = train.inputs[:, perm[k:(k + batch - 1)]]
        ℓ, back = Zygote.pullback(p -> loss(fpipe, Pb, fproxy(Pb, p)), fps)
        global fopt, fps = Optimisers.update(fopt, fps, back(one(ℓ))[1])
    end
end
P = test.inputs
fpred = complete(fpipe, P, fproxy(P, fps))
frep = evaluate(fpipe, P, fpred)
(;
    mean_relative_gap=mean((frep.objective .- test.objective) ./ abs.(test.objective)),
    max_violation=maximum(frep.max_violation),
)
