# Comparisons with solver optima on small LP/QP/NLP instances.

# Gradient descent with momentum on the free per-instance outputs (a
# "proxy" with one free column per instance): a deterministic stand-in for
# training that isolates the method from the network.
function minimize_outputs(pipe, P, O; iters, lr, β=0.9)
    O = copy(O)
    V = zero(O)
    B = size(O, 2)
    for _ in 1:iters
        g = Zygote.gradient(o -> loss(pipe, P, o), O)[1] .* B
        V = β .* V .+ g
        O = O .- lr .* V
    end
    return O
end

relgap(f, fstar) = abs.(f .- fstar) ./ max.(1, abs.(fstar))

@testset "QP (Section 4.1 of Donti et al.) vs HiGHS" begin
    model, _ = dc3_qp(; n=10, neq=4, nin=5, seed=11)
    prob = ParametricProblem(model)
    rng = StableRNG(21)
    P = uniform(rng, 4, 6)
    Ystar, fstar = solve_all(prob, P, HiGHS.Optimizer)
    pipe = L2OPipeline(prob, DC3(; train_steps=5, test_steps=200, test_tol=1e-9))
    d = pipe.decomposition
    Zi = independent_indices(d)
    # At the optimum: completion and correction reproduce y*, the soft loss is f*.
    @test complete(pipe, P, Ystar[Zi, :]).primal ≈ Ystar atol = 1e-6
    @test loss(pipe, P, Ystar[Zi, :]) ≈ sum(fstar) / 6 atol = 1e-6
    # Minimizing the DC3 loss over the independent variables approaches f*
    # (step scaled to the curvature λ/γ of the penalty term).
    Z = minimize_outputs(pipe, P, zeros(6, 6); iters=1000, lr=0.5 * d.step_size / 10)
    pred = complete(pipe, P, Z)
    @test max_equality_violation(prob, P, pred) <= 1e-9
    @test max_violation(prob, P, pred) <= 1e-6
    f = batch_objective(d.form, pred.primal, P)
    @test all(relgap(f, fstar) .< 1e-2)
    @test all(f .>= fstar .- 1e-6)   # feasible points cannot beat the optimum
end

@testset "LP (min-cost flow) vs HiGHS" begin
    prob = ParametricProblem(flow_lp())
    rng = StableRNG(22)
    P = uniform(rng, 2, 5; lo=0.5, hi=2.5)
    Ystar, fstar = solve_all(prob, P, HiGHS.Optimizer)
    pipe = L2OPipeline(prob, DC3(; test_steps=500, test_tol=1e-9))
    Zi = independent_indices(pipe.decomposition)
    @test complete(pipe, P, Ystar[Zi, :]).primal ≈ Ystar atol = 1e-7
    γ = pipe.decomposition.step_size
    Z = minimize_outputs(pipe, P, ones(2, 5); iters=1000, lr=0.5 * γ / 10)
    pred = complete(pipe, P, Z)
    @test max_violation(prob, P, pred) <= 1e-6
    f = batch_objective(pipe.decomposition.form, pred.primal, P)
    @test all(relgap(f, fstar) .< 2e-2)
    @test all(f .>= fstar .- 1e-6)
end

@testset "non-convex QP (Section 4.2 of Donti et al.) vs Ipopt" begin
    model, data = dc3_qp(; n=8, neq=3, nin=4, seed=12, objective=:sine)
    prob = ParametricProblem(model)
    rng = StableRNG(23)
    P = uniform(rng, 3, 4)
    # As in the paper, IPOPT is started at the feasible point A⁺x.
    ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
    Ystar, fstar = similar(P, 8, 4), zeros(4)
    for k in 1:4
        set_start_value.(model[:y], pinv(data.A) * P[:, k])
        Yk, fk = solve_all(prob, P[:, k:k], ipopt)
        Ystar[:, k], fstar[k] = Yk[:, 1], fk[1]
    end
    pipe = L2OPipeline(prob, DC3(; completion=:linear, step_size=1e-2, test_steps=0))
    @test pipe.decomposition.form isa NLPForm   # sin objective: MOI.Nonlinear form
    Zi = independent_indices(pipe.decomposition)
    @test complete(pipe, P, Ystar[Zi, :]).primal ≈ Ystar atol = 1e-7
    # Local descent from the solver's point keeps the objective at a local optimum.
    Z = minimize_outputs(pipe, P, Ystar[Zi, :]; iters=50, lr=1e-3)
    f = batch_objective(pipe.decomposition.form, complete(pipe, P, Z).primal, P)
    @test all(relgap(f, fstar) .< 1e-3)
end

@testset "smooth NLP (nonlinear equalities) vs Ipopt" begin
    model = smooth_nlp()
    prob = ParametricProblem(model)
    rng = StableRNG(24)
    P = uniform(rng, 2, 3; lo=0.2, hi=0.8)
    ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0, "tol" => 1e-10)
    Ystar, fstar = solve_all(prob, P, ipopt)
    pipe = L2OPipeline(
        prob, DC3(; step_size=0.05, train_steps=3, test_steps=100, test_tol=1e-10)
    )
    d = pipe.decomposition
    Zi = independent_indices(d)
    pred = complete(pipe, P, Ystar[Zi, :])
    @test pred.primal ≈ Ystar atol = 1e-6
    @test max_equality_violation(prob, P, pred; evaluator=d.form) <= 1e-9
    # No inequality is active at these optima, so y* is a stationary point of
    # the DC3 loss (penalties and corrections vanish nearby) ...
    @test maximum(abs, Zygote.gradient(z -> loss(pipe, P, z), Ystar[Zi, :])[1]) < 1e-8
    # ... and descent from a perturbed start returns to it. (The reduced
    # objective is nonconvex: from far away descent can stop at another local
    # minimum, which is expected.)
    Z = minimize_outputs(pipe, P, Ystar[Zi, :] .+ 0.05; iters=500, lr=0.01)
    pred = complete(pipe, P, Z)
    f = batch_objective(d.form, pred.primal, P)
    @test max_violation(prob, P, pred; evaluator=d.form) <= 1e-6
    @test all(relgap(f, fstar) .< 1e-6)
end
