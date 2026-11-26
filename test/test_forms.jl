@testset "AffineForm matches L2OCore evaluators" begin
    for sense in (MOI.MIN_SENSE, MOI.MAX_SENSE)
        prob = ParametricProblem(flow_lp(; sense=sense))
        form = AffineForm(prob)
        rng = StableRNG(3)
        Y, P = randn(rng, 4, 5), uniform(rng, 2, 5; lo=0, hi=3)
        mev, jev = MatrixEvaluator(prob), JuMPEvaluator(prob)
        @test batch_objective(form, Y, P) ≈ batch_objective(jev, Y, P)
        @test constraint_values(form, Y, P) ≈ constraint_values(jev, Y, P)
        @test row_violation(form, Y, P) ≈ row_violation(mev, Y, P)
        @test equality_rows(form) == equality_rows(mev)
        @test min_objective(form, Y, P) ≈ objective_sign(form) .* batch_objective(jev, Y, P)
        @test objective_sign(form) == (sense == MOI.MIN_SENSE ? 1 : -1)
        @test L2OCore.num_decision_variables(form) == 4
    end
    model, data = dc3_qp()
    prob = ParametricProblem(model)
    form = problem_form(prob)
    @test form isa AffineForm
    rng = StableRNG(4)
    Y, P = randn(rng, 8, 3), uniform(rng, 3, 3)
    @test batch_objective(form, Y, P) ≈ batch_objective(JuMPEvaluator(prob), Y, P)
    # h and ReLU(g) of the paper.
    @test equality_residual(form, Y, P) ≈ data.A * Y .- P
    @test inequality_residual(form, Y, P) ≈ max.(data.G * Y .- data.h, 0)
    @test inequality_violation(form, Y, P) ≈
        vec(sum(max.(data.G * Y .- data.h, 0) .^ 2; dims=1))
    @test equality_violation(form, Y, P) ≈ vec(sum((data.A * Y .- P) .^ 2; dims=1))
    @test total_violation(form, Y, P) ≈
        equality_violation(form, Y, P) .+ inequality_violation(form, Y, P)
end

@testset "Lipschitz constant of the violation gradient" begin
    model, _ = dc3_qp(; seed=5)
    prob = ParametricProblem(model)
    form = AffineForm(prob)
    L = lipschitz_constant(form)
    @test L ≈ 2 * opnorm(form.A)^2
    rng = StableRNG(6)
    P = uniform(rng, 3, 1)
    # ‖∇φ(a) - ∇φ(b)‖ ≤ L ‖a - b‖ on random pairs.
    for _ in 1:20
        a, b = 3 .* randn(rng, 8, 1), 3 .* randn(rng, 8, 1)
        ga, gb = violation_gradient(form, a, P), violation_gradient(form, b, P)
        @test norm(ga - gb) <= L * norm(a - b) * (1 + 1e-12)
    end
    @test_throws ArgumentError lipschitz_constant(form; rows=:foo)
    @test lipschitz_constant(form; rows=:equality) <= L
end

@testset "NLPForm matches JuMPEvaluator" begin
    for model in (smooth_nlp(), first(dc3_qp(; objective=:sine)), flow_lp())
        prob = ParametricProblem(model)
        form = NLPForm(prob)
        n, np = L2OCore.num_decision_variables(form), num_parameters(prob)
        rng = StableRNG(7)
        Y, P = 0.5 .* randn(rng, n, 4), uniform(rng, np, 4; lo=0, hi=1)
        jev = JuMPEvaluator(prob)
        @test batch_objective(form, Y, P) ≈ batch_objective(jev, Y, P)
        @test constraint_values(form, Y, P) ≈ constraint_values(jev, Y, P)
        @test row_violation(form, Y, P) ≈ row_violation(jev, Y, P)
        @test equality_rows(form) == equality_rows(jev)
        @test L2OCore.bound_rows(form) == L2OCore.bound_rows(jev)
    end
    @test problem_form(ParametricProblem(smooth_nlp())) isa NLPForm
    @test_throws DimensionMismatch batch_objective(
        NLPForm(ParametricProblem(smooth_nlp())), zeros(3, 1), zeros(2, 1)
    )
end

@testset "NLPForm derivatives (rrules) match finite differences" begin
    prob = ParametricProblem(smooth_nlp())
    form = NLPForm(prob)
    rng = StableRNG(8)
    Y, P = 0.5 .* randn(rng, 4, 3), uniform(rng, 2, 3; lo=0, hi=1)
    m = L2OCore.num_duals(prob)
    W, V, w = randn(rng, m, 3), randn(rng, 4, 3), randn(rng, 3)
    @test grad_error(y -> sum(batch_objective(form, y, P) .* w), Y) < 1e-8
    @test grad_error(p -> sum(batch_objective(form, Y, p) .* w), P) < 1e-8
    @test grad_error(y -> sum(constraint_values(form, y, P) .* W), Y) < 1e-8
    @test grad_error(p -> sum(constraint_values(form, Y, p) .* W), P) < 1e-8
    # constraint_vjp needs second derivatives (Hessian-of-Lagrangian products).
    @test grad_error(y -> sum(constraint_vjp(form, y, P, W) .* V), Y) < 1e-8
    @test grad_error(p -> sum(constraint_vjp(form, Y, p, W) .* V), P) < 1e-8
    @test grad_error(w -> sum(constraint_vjp(form, Y, P, w) .* V), W) < 1e-8
    # The analytic Jacobian.
    J = constraint_jacobian(form, Y[:, 1], P[:, 1])
    Jfd = FiniteDifferences.jacobian(
        FiniteDifferences.central_fdm(5, 1),
        x -> vec(constraint_values(form, reshape(x[1:4], :, 1), reshape(x[5:6], :, 1))),
        vcat(Y[:, 1], P[:, 1]),
    )[1]
    @test J ≈ Jfd atol = 1e-8
end

@testset "unsupported problems" begin
    prob = ParametricProblem(psd_problem())
    @test_throws ArgumentError AffineForm(prob)
    @test_throws ArgumentError NLPForm(prob)
    @test_throws ArgumentError L2OPipeline(prob, DC3())
end
