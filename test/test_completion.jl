@testset "linear completion satisfies the equalities; IFT Jacobian" begin
    for seed in 1:3
        model, data = dc3_qp(; n=10, neq=4, nin=5, seed=seed)
        prob = ParametricProblem(model)
        pipe = L2OPipeline(prob, DC3(; test_steps=0))
        d = pipe.decomposition
        c = d.completion
        @test c isa LinearCompletion
        @test output_dimension(pipe) == 6
        @test sort(vcat(independent_indices(c), dependent_indices(c))) == 1:10
        rng = StableRNG(seed)
        P, Z = uniform(rng, 4, 7), randn(rng, 6, 7)
        Y = equality_completion(c, Z, P)
        @test Y[independent_indices(c), :] == Z
        # The completion satisfies the equalities: h_x(φ_x(z)) = 0.
        @test max_equality_violation(prob, P, Prediction(; primal=Y)) <= 1e-10
        @test maximum(abs, data.A * Y .- P) < 1e-10
        # eq. (3): ∂φ/∂z = -(J^h_φ)⁻¹ J^h_z.
        D, Zi = dependent_indices(c), independent_indices(c)
        @test c.Kz[D, :] ≈ -(data.A[:, D] \ data.A[:, Zi])
        # IFT Jacobian-vector products vs finite differences.
        @test maximum(completion_jacobian_error(pipe, P, Z; rng=StableRNG(1))) <= 1e-5
        @test grad_error(z -> sum(equality_completion(c, z, P) .^ 2), Z) < 1e-8
    end
end

@testset "completion is the identity on feasible points" begin
    # For any feasible y, completing its independent part gives back y.
    model, data = dc3_qp(; seed=2)
    prob = ParametricProblem(model)
    c = L2OPipeline(prob, DC3()).decomposition.completion
    rng = StableRNG(9)
    P = uniform(rng, 3, 4)
    Y = pinv(data.A) * P .+ nullspace(data.A) * randn(rng, 5, 4)
    @test equality_completion(c, Y[independent_indices(c), :], P) ≈ Y
end

@testset "choice of dependent variables" begin
    model, data = dc3_qp(; n=6, neq=2, nin=2, seed=4)
    prob = ParametricProblem(model)
    # QR with column pivoting picks a nonsingular block.
    rows, D = select_dependent(data.A)
    @test rows == [1, 2] && length(D) == 2
    @test rank(data.A[:, D]) == 2
    @test select_dependent(zeros(0, 3)) == (Int[], Int[])
    # User override.
    pipe = L2OPipeline(prob, DC3(; dependent=[5, 6]))
    @test dependent_indices(pipe.decomposition) == [5, 6]
    @test independent_indices(pipe.decomposition) == 1:4
    # Wrong number, out of range, singular blocks.
    @test_throws ArgumentError L2OPipeline(prob, DC3(; dependent=[1]))
    @test_throws ArgumentError L2OPipeline(prob, DC3(; dependent=[1, 9]))
    m2 = Model()
    @variable(m2, p in MOI.Parameter(1.0))
    @variable(m2, y[1:3])
    @constraint(m2, y[1] + y[2] == p)
    @objective(m2, Min, sum(y .^ 2))
    @test_throws ArgumentError L2OPipeline(ParametricProblem(m2), DC3(; dependent=[3]))
end

@testset "redundant equalities" begin
    prob = ParametricProblem(redundant_lp())
    pipe = L2OPipeline(prob, DC3())
    @test output_dimension(pipe) == 2   # one equality is redundant
    P = reshape([0.3, -1.0], 1, 2)
    Y = equality_completion(pipe.decomposition.completion, [0.1 0.2; -0.4 0.5], P)
    @test max_equality_violation(prob, P, Prediction(; primal=Y)) <= 1e-12
    # Inconsistent redundancy: infeasible for some parameters.
    @test_throws ArgumentError L2OPipeline(
        ParametricProblem(redundant_lp(; consistent=false)), DC3()
    )
end

@testset "problems without equalities" begin
    m = Model()
    @variable(m, p in MOI.Parameter(1.0))
    @variable(m, -1 <= y[1:2] <= 1)
    @objective(m, Min, (y[1] - p)^2 + y[2]^2)
    pipe = L2OPipeline(ParametricProblem(m), DC3())
    @test output_dimension(pipe) == 2
    @test pipe.decomposition.completion.Kz == I(2)
end

@testset "Newton completion (nonlinear equalities)" begin
    prob = ParametricProblem(smooth_nlp())
    pipe = L2OPipeline(prob, DC3(; step_size=0.05, test_steps=0))
    d = pipe.decomposition
    c = d.completion
    @test c isa NewtonCompletion
    @test d.form isa NLPForm
    rng = StableRNG(10)
    P, Z = uniform(rng, 2, 6; lo=0, hi=1), 0.6 .* randn(rng, 2, 6)
    Y = equality_completion(c, Z, P)
    @test max_equality_violation(prob, P, Prediction(; primal=Y); evaluator=d.form) <= 1e-9
    @test maximum(completion_jacobian_error(pipe, P, Z; rng=StableRNG(2))) <= 1e-5
    # IFT derivatives with respect to z and p, vs finite differences.
    V = randn(rng, 4, 6)
    @test grad_error(z -> sum(equality_completion(c, z, P) .* V), Z) < 1e-7
    @test grad_error(p -> sum(equality_completion(c, Z, p) .* V), P) < 1e-7
    # The pullback used inside the correction, and its own derivatives (second order).
    @test grad_error(v -> sum(completion_pullback(c, Y, P, v) .* Z), V) < 1e-8
    @test grad_error(
        z -> sum(completion_pullback(c, equality_completion(c, z, P), P, V) .* Z), Z
    ) < 1e-6
    # Warm start does not change the root.
    @test equality_completion(c, Z, P; init=Y .+ 0.01) ≈ Y atol = 1e-9
    # Newton on affine equalities: forced via completion = :newton.
    model, _ = dc3_qp(; seed=3)
    pq = ParametricProblem(model)
    pn = L2OPipeline(pq, DC3(; completion=:newton, step_size=1e-2, test_steps=0))
    pl = L2OPipeline(pq, DC3(; test_steps=0, dependent=dependent_indices(pn.decomposition)))
    P2, Z2 = uniform(rng, 3, 4), randn(rng, 5, 4)
    @test complete(pn, P2, Z2).primal ≈ complete(pl, P2, Z2).primal atol = 1e-9
    @test_throws ArgumentError L2OPipeline(prob, DC3(; completion=:linear, step_size=0.1))
    @test_throws ArgumentError DC3(; completion=:foo)
end
