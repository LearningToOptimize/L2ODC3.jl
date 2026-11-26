# Second-order cones through the projection residual F - Π_K(F).

function cone_points(rng, d, N)
    F = randn(rng, d, N)
    F[1, 1:3:end] .= 3 .* sqrt.(sum(abs2, F[2:end, 1:3:end]; dims=1))[:]    # inside
    F[1, 2:3:end] .= -3 .* sqrt.(sum(abs2, F[2:end, 2:3:end]; dims=1))[:]   # polar
    return F                                                                # mixed
end

@testset "SOC projection residual (Moreau decomposition)" begin
    rng = StableRNG(51)
    for d in (2, 4, 7)
        F = cone_points(rng, d, 30)
        R = soc_residual(F)
        Π = F .- R
        K = MOI.SecondOrderCone(d)
        for k in axes(F, 2)
            # ‖F - Π(F)‖ = dist(F, K); Π(F) ∈ K; -r ∈ K (K self-dual); ⟨Π(F), r⟩ = 0.
            @test norm(R[:, k]) ≈ MOI.Utilities.distance_to_set(F[:, k], K) atol = 1e-12
            @test MOI.Utilities.distance_to_set(Π[:, k], K) < 1e-12
            @test MOI.Utilities.distance_to_set(-R[:, k], K) < 1e-12
            @test abs(dot(Π[:, k], R[:, k])) < 1e-12
        end
        @test maximum(abs, soc_residual(Π)) < 1e-12           # idempotent projection
        # ∇ dist²(F, K) = 2(F - Π_K(F)).
        G = Zygote.gradient(F -> sum(abs2, soc_residual(F)), F)[1]
        @test G ≈ 2 .* R atol = 1e-10
    end
    # Finite derivative at x = 0 (smoothed norm).
    F0 = [1.0 -1.0; 0.0 0.0; 0.0 0.0]
    @test soc_residual(F0) ≈ [0.0 -1.0; 0.0 0.0; 0.0 0.0]
    @test all(isfinite, Zygote.gradient(F -> sum(abs2, soc_residual(F)), F0)[1])
end

@testset "rotated SOC projection residual" begin
    rng = StableRNG(52)
    F = randn(rng, 5, 40)
    F[1:2, 1:2:end] .= abs.(F[1:2, 1:2:end]) .+ 2      # many points inside
    R = rsoc_residual(F)
    K = MOI.RotatedSecondOrderCone(5)
    # MOI's distance for this cone is a violation measure, not the Euclidean
    # distance (and it is 1 at the origin), so check the Moreau characterization
    # of the projection with an explicit membership test: Π(F) ∈ K,
    # -r ∈ K* = K (self-dual) and ⟨Π(F), r⟩ = 0 determine Π uniquely.
    in_rsoc(v; tol=1e-10) =
        v[1] >= -tol && v[2] >= -tol && 2 * v[1] * v[2] - sum(abs2, v[3:end]) >= -tol
    for k in axes(F, 2)
        @test in_rsoc(F[:, k] .- R[:, k])
        @test in_rsoc(-R[:, k])
        @test abs(dot(F[:, k] .- R[:, k], R[:, k])) < 1e-10
    end
    @test maximum(abs, rsoc_residual(F .- R)) < 1e-10
    G = Zygote.gradient(F -> sum(abs2, rsoc_residual(F)), F)[1]
    @test G ≈ 2 .* R atol = 1e-10
end

@testset "forms with cones" begin
    for model in (portfolio_socp(), rsoc_problem(), socp())
        prob = ParametricProblem(model)
        form = problem_form(prob)
        @test form isa AffineForm
        @test length(cone_blocks(form)) == 1
        nlp = NLPForm(prob)
        @test cone_blocks(nlp) == cone_blocks(form)
        @test violation_indices(nlp) == violation_indices(form)
        n, np = L2OCore.num_decision_variables(form), num_parameters(prob)
        rng = StableRNG(53)
        Y, P = randn(rng, n, 5), uniform(rng, np, 5; lo=0.2, hi=1)
        mev = MatrixEvaluator(prob)
        @test cone_violation(nlp, Y, P) ≈ cone_violation(form, Y, P) atol = 1e-12
        @test total_violation(nlp, Y, P) ≈ total_violation(form, Y, P)
        # total violation = squared row violations + squared cone distances.
        @test total_violation(form, Y, P) ≈
            vec(sum(abs2, row_violation(mev, Y, P); dims=1)) .+
              vec(sum(abs2, cone_violation(form, Y, P); dims=1))
        if last(only(cone_blocks(form))) === :soc
            # MOI's distance is Euclidean for the (unrotated) second-order cone.
            @test cone_violation(form, Y, P) ≈ cone_violation(mev, Y, P) atol = 1e-12
        end
        @test grad_error(y -> sum(total_violation(form, y, P)), Y) < 1e-7
        @test grad_error(y -> sum(total_violation(nlp, y, P)), Y) < 1e-7
        # L = 2‖A_R‖² still bounds the gradient's Lipschitz constant with cones.
        L = lipschitz_constant(form)
        for _ in 1:10
            a, b = 2 .* randn(rng, n, 1), 2 .* randn(rng, n, 1)
            ga, gb = violation_gradient(form, a, P[:, 1:1]),
            violation_gradient(form, b, P[:, 1:1])
            @test norm(ga - gb) <= L * norm(a - b) * (1 + 1e-12)
        end
    end
end

@testset "DC3 on a portfolio SOCP vs Clarabel" begin
    model = portfolio_socp(; seed=54)
    prob = ParametricProblem(model)
    rng = StableRNG(54)
    P = vcat(rand(rng, 6, 5), fill(0.5, 1, 5))
    clarabel = optimizer_with_attributes(Clarabel.Optimizer, "verbose" => false)
    Ystar, fstar = solve_all(prob, P, clarabel)
    pipe = L2OPipeline(prob, DC3(; test_steps=500, test_tol=1e-9))
    d = pipe.decomposition
    Zi = independent_indices(d)
    # Completion and correction keep the (interior-point) optimum.
    @test complete(pipe, P, Ystar[Zi, :]).primal ≈ Ystar atol = 1e-6
    # Descent and derivative checks with a cone among the inequalities.
    Z0 = randn(rng, 5, 5)
    @test maximum(correction_descent_error(pipe, P, Z0; steps=50)) <= 1e-8
    @test maximum(completion_jacobian_error(pipe, P, Z0; rng=StableRNG(1))) <= 1e-5
    _, Yc = inequality_correction(
        d.form, d.completion, Z0, P; steps=5000, step_size=d.step_size
    )
    @test maximum(inequality_violation(d.form, Yc, P)) < 1e-12
    @test grad_error(z -> loss(pipe, P, z), Z0) < 1e-7
    # Descent on the DC3 loss over free outputs approaches the optimum.
    Z = minimize_outputs(pipe, P, zeros(5, 5); iters=1500, lr=0.5 * d.step_size / 10)
    pred = complete(pipe, P, Z)
    @test max_violation(prob, P, pred) <= 1e-6
    f = batch_objective(d.form, pred.primal, P)
    @test all(relgap(f, fstar) .< 2e-2)
    @test all(f .>= fstar .- 1e-6)
    # Same predictions for the maximization form.
    pmax = L2OPipeline(
        ParametricProblem(portfolio_socp(; seed=54, sense=MOI.MAX_SENSE)), DC3()
    )
    @test complete(pmax, P, Z0).primal ≈ complete(L2OPipeline(prob, DC3()), P, Z0).primal
end

@testset "FSNet with cones" begin
    for model in (portfolio_socp(; seed=55), rsoc_problem())
        prob = ParametricProblem(model)
        pipe = L2OPipeline(prob, FSNet(; train_steps=4))
        d = pipe.decomposition
        n, np = L2OCore.num_decision_variables(d.form), num_parameters(prob)
        rng = StableRNG(55)
        P = uniform(rng, np, 4; lo=0.3, hi=1)
        Y = 2 .* randn(rng, n, 4)
        @test maximum(feasibility_descent_error(pipe, P, Y; steps=50)) <= 1e-8
        S = feasibility_seeking(d.form, Y, P; steps=20_000, step_size=d.step_size)
        @test max_violation(prob, P, Prediction(; primal=S)) <= 1e-6
        @test grad_error(y -> loss(pipe, P, y), Y) < 1e-7
    end
end

@testset "cones on JLArray" begin
    model = portfolio_socp(; seed=56)
    method = DC3(; train_steps=3, test_steps=5)
    pipe = L2OPipeline(ParametricProblem(model), method)
    d = pipe.decomposition
    dj = Adapt.adapt(JLArray, d)
    rng = StableRNG(56)
    P, Z = vcat(rand(rng, 6, 4), fill(0.5, 1, 4)), randn(rng, 5, 4)
    Pj, Zj = JLArray(P), JLArray(Z)
    @test Array(predict(method, dj, Pj, Zj)) ≈ predict(method, d, P, Z)
    gj = Zygote.gradient(z -> training_loss(method, dj, Pj, z), Zj)[1]
    @test Array(gj) ≈ Zygote.gradient(z -> training_loss(method, d, P, z), Z)[1]
end
