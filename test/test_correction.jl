@testset "correction steps decrease the violation (affine constraints, step 1/L)" begin
    for seed in 1:3
        model, _ = dc3_qp(; n=10, neq=4, nin=6, seed=seed)
        prob = ParametricProblem(model)
        pipe = L2OPipeline(prob, DC3())
        rng = StableRNG(seed)
        P, Z = uniform(rng, 4, 8), 3 .* randn(rng, 6, 8)
        @test maximum(correction_descent_error(pipe, P, Z; steps=50)) <= 1e-8
    end
    prob = ParametricProblem(flow_lp())
    pipe = L2OPipeline(prob, DC3())
    rng = StableRNG(11)
    P, Z = uniform(rng, 2, 5; lo=0, hi=3), 5 .* randn(rng, 2, 5)
    @test maximum(correction_descent_error(pipe, P, Z)) <= 1e-8
end

@testset "correction converges to feasibility (affine, feasible instances)" begin
    # Section 3.2 of Donti et al.: for linear constraints the correction
    # converges. The paper's QP instances are feasible by
    # construction (eq. (6)), so the violation must vanish.
    model, _ = dc3_qp(; n=10, neq=4, nin=6, seed=7)
    prob = ParametricProblem(model)
    d = L2OPipeline(prob, DC3()).decomposition
    rng = StableRNG(12)
    P, Z = uniform(rng, 4, 6), 3 .* randn(rng, 6, 6)
    Y0 = equality_completion(d.completion, Z, P)
    @test maximum(inequality_violation(d.form, Y0, P)) > 1e-1
    _, Y = inequality_correction(
        d.form, d.completion, Z, P; steps=3000, step_size=d.step_size
    )
    @test maximum(inequality_violation(d.form, Y, P)) < 1e-10
    @test maximum(abs, equality_residual(d.form, Y, P)) < 1e-10   # equalities kept
    # Early stopping at test time.
    _, Yt = inequality_correction(
        d.form, d.completion, Z, P; steps=10_000, step_size=d.step_size, tol=1e-6
    )
    @test maximum(abs, inequality_residual(d.form, Yt, P)) <= 1e-6
end

@testset "correction is a no-op at feasible points" begin
    model, data = dc3_qp(; seed=8)
    prob = ParametricProblem(model)
    pipe = L2OPipeline(prob, DC3(; test_steps=25))
    rng = StableRNG(13)
    P = uniform(rng, 3, 3)
    Y = pinv(data.A) * P            # feasible by eq. (6)
    Zi = independent_indices(pipe.decomposition)
    @test complete(pipe, P, Y[Zi, :]).primal ≈ Y
end

@testset "gradients through unrolled corrections" begin
    # Affine: momentum 0.5 (paper default), several steps.
    model, _ = dc3_qp(; seed=9)
    prob = ParametricProblem(model)
    pipe = L2OPipeline(prob, DC3(; train_steps=5, momentum=0.5))
    rng = StableRNG(14)
    P, Z = uniform(rng, 3, 4), 2 .* randn(rng, 5, 4)
    @test grad_error(z -> loss(pipe, P, z), Z) < 1e-7
    # Reduced gradient = gradient of the reduced violation v(z).
    d = pipe.decomposition
    v(z) = sum(inequality_violation(d.form, equality_completion(d.completion, z, P), P))
    G = reduced_violation_gradient(
        d.form, d.completion, equality_completion(d.completion, Z, P), P
    )
    @test G ≈ Zygote.gradient(v, Z)[1]
    # Nonlinear: the loss differentiates Newton completions and second-order terms.
    pn = L2OPipeline(ParametricProblem(smooth_nlp()), DC3(; step_size=0.05, train_steps=3))
    Pn, Zn = uniform(rng, 2, 3; lo=0, hi=1), 0.8 .* randn(rng, 2, 3)
    @test grad_error(z -> loss(pn, Pn, z), Zn) < 1e-6
    dn = pn.decomposition
    vn(z) =
        sum(inequality_violation(dn.form, equality_completion(dn.completion, z, Pn), Pn))
    Gn = reduced_violation_gradient(
        dn.form, dn.completion, equality_completion(dn.completion, 2 .* Zn, Pn), Pn
    )
    @test Gn ≈ Zygote.gradient(vn, 2 .* Zn)[1] rtol = 1e-8
    @test grad_error(
        z -> sum(
            reduced_violation_gradient(
                dn.form, dn.completion, equality_completion(dn.completion, z, Pn), Pn
            ),
        ),
        2 .* Zn,
    ) < 1e-6
end

@testset "nonlinear correction keeps the equalities" begin
    prob = ParametricProblem(smooth_nlp())
    pipe = L2OPipeline(prob, DC3(; step_size=0.05, test_steps=30, momentum=0.0))
    rng = StableRNG(15)
    P, Z = uniform(rng, 2, 5; lo=0, hi=1), 1.2 .* randn(rng, 2, 5)
    pred = complete(pipe, P, Z)
    @test max_equality_violation(prob, P, pred; evaluator=pipe.decomposition.form) <= 1e-9
    d = pipe.decomposition
    @test sum(inequality_violation(d.form, pred.primal, P)) <
        sum(inequality_violation(d.form, equality_completion(d.completion, Z, P), P))
end

@testset "FSNet feasibility seeking" begin
    model, _ = dc3_qp(; n=10, neq=4, nin=6, seed=10)
    prob = ParametricProblem(model)
    pipe = L2OPipeline(prob, FSNet())
    d = pipe.decomposition
    @test d.step_size ≈ 1 / lipschitz_constant(d.form)
    rng = StableRNG(16)
    P, Y = uniform(rng, 4, 6), 3 .* randn(rng, 10, 6)
    @test maximum(feasibility_descent_error(pipe, P, Y; steps=50)) <= 1e-8
    # Theorem 1 of Nguyen & Donti (PL holds for affine constraints): linear
    # convergence of φ to zero on feasible instances.
    φ0 = total_violation(d.form, Y, P)
    S = feasibility_seeking(d.form, Y, P; steps=5000, step_size=d.step_size)
    @test maximum(total_violation(d.form, S, P)) < 1e-12 * max(1, maximum(φ0))
    # Truncated unrolling: with unroll = 0 the map has the identity derivative.
    V = randn(rng, 10, 6)
    g0 = Zygote.gradient(
        y -> sum(
            feasibility_seeking(d.form, y, P; steps=5, unroll=0, step_size=d.step_size) .* V,
        ),
        Y,
    )[1]
    @test g0 ≈ V
    # unroll = K' < K: derivative of the first K' steps.
    gK = Zygote.gradient(
        y -> sum(
            feasibility_seeking(d.form, y, P; steps=7, unroll=3, step_size=d.step_size) .* V,
        ),
        Y,
    )[1]
    g3 = Zygote.gradient(
        y -> sum(feasibility_seeking(d.form, y, P; steps=3, step_size=d.step_size) .* V), Y
    )[1]
    @test gK ≈ g3
    @test feasibility_seeking(d.form, Y, P; steps=7, unroll=3, step_size=d.step_size) ≈
        feasibility_seeking(d.form, Y, P; steps=7, step_size=d.step_size)
    # Full unrolled gradient vs finite differences, and the FSNet loss.
    @test grad_error(
        y -> sum(feasibility_seeking(d.form, y, P; steps=4, step_size=d.step_size) .* V), Y
    ) < 1e-7
    @test grad_error(y -> loss(pipe, P, y), Y) < 1e-7
    # Early stop.
    Se = feasibility_seeking(d.form, Y, P; steps=10^5, step_size=d.step_size, tol=1e-8)
    @test maximum(total_violation(d.form, Se, P)) <= 1e-8
    @test_throws ArgumentError feasibility_seeking(
        d.form, Y, P; steps=2, unroll=-1, step_size=1.0
    )
end

@testset "FSNet on a nonlinear problem" begin
    prob = ParametricProblem(smooth_nlp())
    @test_throws ArgumentError L2OPipeline(prob, FSNet())   # no default step size
    pipe = L2OPipeline(
        prob, FSNet(; step_size=0.05, train_steps=3, penalty=1.0, penalty_threshold=0.0)
    )
    rng = StableRNG(17)
    P, Y = uniform(rng, 2, 3; lo=0, hi=1), 0.8 .* randn(rng, 4, 3)
    @test maximum(feasibility_descent_error(pipe, P, Y; steps=10)) <= 1e-8
    @test grad_error(y -> loss(pipe, P, y), Y) < 1e-6
end
