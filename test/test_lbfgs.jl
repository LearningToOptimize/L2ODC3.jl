# L-BFGS inner solver for FSNet (the solver of the experiments of Nguyen & Donti).

@testset "L-BFGS feasibility seeking (affine)" begin
    model, _ = dc3_qp(; n=10, neq=4, nin=6, seed=61)
    prob = ParametricProblem(model)
    pipe = L2OPipeline(prob, FSNet(; solver=:lbfgs, train_steps=5))
    d = pipe.decomposition
    rng = StableRNG(61)
    P, Y = uniform(rng, 4, 6), 3 .* randn(rng, 10, 6)
    lb(k; kw...) =
        feasibility_seeking_lbfgs(d.form, Y, P; steps=k, step_size=d.step_size, kw...)
    # Armijo backtracking: φ is non-increasing along the iterates.
    φs = [total_violation(d.form, lb(k), P) for k in 0:15]
    @test all(all(φs[k + 1] .<= φs[k] .* (1 + 1e-12) .+ 1e-30) for k in 1:15)
    # Much faster than gradient descent with step 1/L on the same problem.
    φl = maximum(total_violation(d.form, lb(50), P))
    φg = maximum(
        total_violation(
            d.form, feasibility_seeking(d.form, Y, P; steps=50, step_size=d.step_size), P
        ),
    )
    @test φl < 1e-12
    @test φl < 1e-6 * φg
    # Unrolled derivatives (line-search steps held constant) vs finite differences.
    V = randn(rng, 10, 6)
    @test grad_error(
        y -> sum(
            feasibility_seeking_lbfgs(d.form, y, P; steps=4, step_size=d.step_size) .* V
        ),
        Y,
    ) < 1e-7
    @test grad_error(y -> loss(pipe, P, y), Y) < 1e-7
    # Truncation: unroll = 0 has the identity derivative; unroll = K' < K matches
    # the derivative of the K'-step map; the value is that of the full map.
    g0 = Zygote.gradient(
        y -> sum(
            feasibility_seeking_lbfgs(
                d.form, y, P; steps=6, unroll=0, step_size=d.step_size
            ) .* V,
        ),
        Y,
    )[1]
    @test g0 ≈ V
    gK = Zygote.gradient(
        y -> sum(
            feasibility_seeking_lbfgs(
                d.form, y, P; steps=8, unroll=3, step_size=d.step_size
            ) .* V,
        ),
        Y,
    )[1]
    g3 = Zygote.gradient(
        y -> sum(
            feasibility_seeking_lbfgs(d.form, y, P; steps=3, step_size=d.step_size) .* V
        ),
        Y,
    )[1]
    @test gK ≈ g3
    @test lb(8; unroll=3) ≈ lb(8)
    # Early stop and memory.
    @test maximum(total_violation(d.form, lb(500; tol=1e-14), P)) <= 1e-14
    @test maximum(total_violation(d.form, lb(60; memory=1), P)) < 1e-8
    @test_throws ArgumentError lb(2; memory=0)
    @test_throws ArgumentError lb(2; unroll=-1)
    @test_throws ArgumentError FSNet(; solver=:newton)
    @test_throws ArgumentError FSNet(; memory=0)
    # complete() with L-BFGS reaches feasibility.
    pt = L2OPipeline(prob, FSNet(; solver=:lbfgs, test_steps=100, test_tol=1e-20))
    @test max_violation(prob, P, complete(pt, P, Y)) <= 1e-8
end

@testset "L-BFGS feasibility seeking (nonlinear and cones)" begin
    prob = ParametricProblem(smooth_nlp())
    pipe = L2OPipeline(prob, FSNet(; solver=:lbfgs, train_steps=3))
    @test pipe.decomposition.step_size == 1.0   # first trial step for nonlinear forms
    rng = StableRNG(62)
    P, Y = uniform(rng, 2, 3; lo=0, hi=1), 0.8 .* randn(rng, 4, 3)
    @test grad_error(y -> loss(pipe, P, y), Y) < 1e-6
    S = feasibility_seeking_lbfgs(pipe.decomposition.form, Y, P; steps=100, step_size=1.0)
    @test max_violation(
        prob, P, Prediction(; primal=S); evaluator=pipe.decomposition.form
    ) <= 1e-6
    probc = ParametricProblem(portfolio_socp(; seed=62))
    pc = L2OPipeline(probc, FSNet(; solver=:lbfgs, test_steps=200, test_tol=1e-20))
    Pc, Yc = uniform(rng, 7, 4; lo=0.3, hi=1), 2 .* randn(rng, 6, 4)
    @test max_violation(probc, Pc, complete(pc, Pc, Yc)) <= 1e-7
end

@testset "L-BFGS on JLArray" begin
    model, _ = dc3_qp(; seed=63)
    method = FSNet(; solver=:lbfgs, train_steps=3, unroll=2, test_steps=5)
    pipe = L2OPipeline(ParametricProblem(model), method)
    d = pipe.decomposition
    dj = Adapt.adapt(JLArray, d)
    rng = StableRNG(63)
    P, Y = uniform(rng, 3, 4), randn(rng, 8, 4)
    Pj, Yj = JLArray(P), JLArray(Y)
    @test Array(predict(method, dj, Pj, Yj)) ≈ predict(method, d, P, Y)
    gj = Zygote.gradient(y -> training_loss(method, dj, Pj, y), Yj)[1]
    @test Array(gj) ≈ Zygote.gradient(y -> training_loss(method, d, P, y), Y)[1]
end
