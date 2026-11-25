# Small parametric test problems with known structure.

using JuMP
using LinearAlgebra
using StableRNGs
import MathOptInterface as MOI

"""
Convex QP of Section 4.1 of Donti et al. (eqs. (5)–(6)): `min ½yᵀQy + cᵀy` s.t.
`Ay = x`, `Gy ≤ h`, with `h = Σⱼ |(GA⁺)ᵢⱼ|` so that `A⁺x` is feasible for every
`x ∈ [-1, 1]^neq`. The parameters are `x`.
"""
function dc3_qp(; n=8, neq=3, nin=4, seed=1, objective=:quadratic)
    rng = StableRNG(seed)
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
    if objective === :quadratic
        @objective(model, Min, 0.5 * y' * Q * y + c' * y)
    else
        # The "simple non-convex" variant of Section 4.2: cᵀ sin(y).
        @objective(model, Min, 0.5 * y' * Q * y + sum(c[i] * sin(y[i]) for i in 1:n))
    end
    return model, (; Q, c, A, G, h)
end

"""
Min-cost flow LP with capacity bounds; the demands are parameters. Optionally a
maximization of the negated cost.
"""
function flow_lp(; sense=MOI.MIN_SENSE)
    model = Model()
    @variable(model, d[1:2] in MOI.Parameter.([1.0, 2.0]))
    @variable(model, 0 <= g[1:3] <= 4)
    @variable(model, 0 <= f <= 2.5)
    @constraint(model, g[1] + g[2] - f == d[1])
    @constraint(model, g[3] + f == d[2])
    @constraint(model, g[1] + g[3] <= 5)
    cost = [1.0, 3.0, 2.0]
    if sense == MOI.MIN_SENSE
        @objective(model, Min, sum(cost .* g) + 0.1f)
    else
        @objective(model, Max, -sum(cost .* g) - 0.1f)
    end
    return model
end

"""
Smooth NLP with two nonlinear equalities whose Jacobian block in `(y₃, y₄)` is
unit lower triangular (so the completion is well defined everywhere), a convex
quadratic inequality, bounds, and a nonconvex objective.
"""
function smooth_nlp()
    model = Model()
    @variable(model, p[1:2] in MOI.Parameter.([0.5, 0.4]))
    @variable(model, -2 <= y[1:4] <= 2, start = 0.2)
    @constraint(model, y[3] + 0.3 * y[1]^2 - 0.2 * y[1] * y[2] == p[1])
    @constraint(model, y[4] + 0.2 * exp(0.5 * y[1]) + 0.1 * y[3]^2 == p[2])
    @constraint(model, y[1]^2 + y[2]^2 + 0.5 * y[3]^2 <= 1.0)
    @objective(model, Min, sum((y .- 0.6) .^ 2) + 0.3 * sin(2 * y[2]) + p[1] * y[4])
    return model
end

"Two copies of the same equality (consistent redundancy) plus a box."
function redundant_lp(; consistent=true)
    model = Model()
    @variable(model, p in MOI.Parameter(1.0))
    @variable(model, -3 <= y[1:3] <= 3)
    @constraint(model, y[1] + y[2] + y[3] == p)
    @constraint(model, 2y[1] + 2y[2] + 2y[3] == (consistent ? 2p : p))
    @constraint(model, y[1] - y[2] <= 1)
    @objective(model, Min, y[1]^2 + y[2]^2 + y[3]^2)
    return model
end

"Second-order cone constraint: not supported by DC3/FSNet forms."
function socp()
    model = Model()
    @variable(model, p in MOI.Parameter(1.0))
    @variable(model, t)
    @variable(model, x[1:2])
    @constraint(model, x[1] + x[2] == p)
    @constraint(model, [t; x] in SecondOrderCone())
    @objective(model, Min, t)
    return model
end

"Binary variable: MixedInteger, not supported."
function milp()
    model = Model()
    @variable(model, p in MOI.Parameter(1.0))
    @variable(model, z, Bin)
    @variable(model, x >= 0)
    @constraint(model, x + z == p)
    @objective(model, Min, x + 2z)
    return model
end

"Uniform samples in `[lo, hi]` (`dim × N`) from a seeded RNG."
uniform(rng, dim, N; lo=-1.0, hi=1.0) = lo .+ (hi - lo) .* rand(rng, dim, N)

"Solve every column of `P` and return `(Y*, f*)`."
function solve_all(problem, P, optimizer)
    ds = L2OCore.solve_instances(problem, P; optimizer=optimizer, primal=true)
    @assert all(in((MOI.OPTIMAL, MOI.LOCALLY_SOLVED)), ds.status) ds.status
    return ds.primal, ds.objective
end

"Zygote gradient vs. central finite differences (relative max error)."
function grad_error(f, x)
    x0 = copy(x)  # FiniteDifferences perturbs its argument in place
    g = Zygote.gradient(f, copy(x0))[1]
    gfd = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), f, copy(x0))[1]
    return maximum(abs, g .- gfd) / max(1, maximum(abs, gfd))
end
