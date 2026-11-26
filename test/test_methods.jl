@testset "method traits" begin
    @test supports(DC3(), LinearConic())
    @test supports(DC3(), ConvexQuadratic())
    @test supports(DC3(), SmoothNonlinear())
    @test !supports(DC3(), MixedInteger())
    @test supports(FSNet(), SmoothNonlinear())
    @test !supports(FSNet(), MixedInteger())
    @test method_family(DC3()) == PrimalFamily()
    @test_throws UnsupportedProblemError L2OPipeline(ParametricProblem(milp()), DC3())
    @test_throws ArgumentError DC3(; step_size=-1.0)
    @test_throws ArgumentError DC3(; momentum=1.0)
    @test_throws ArgumentError DC3(; train_steps=-1)
    @test_throws ArgumentError FSNet(; step_size=0.0)
    @test_throws ArgumentError FSNet(; unroll=-1)
end

@testset "pipeline: DC3 on the paper's QP" begin
    model, _ = dc3_qp()
    prob = ParametricProblem(model)
    pipe = L2OPipeline(prob, DC3())
    @test pipe.decomposition isa DC3Decomposition
    @test input_dimension(pipe) == 3
    @test output_dimension(pipe) == 5
    # Self-supervised data: no solver calls without an optimizer.
    ds = generate_data(pipe; sampler=BoxSampler(-ones(3), ones(3)), n=10, rng=StableRNG(1))
    @test num_samples(ds) == 10
    @test ds.primal === nothing && ds.status === nothing
    @test ds.metadata["method"] == "DC3"
    @test ds.metadata["independent"] == independent_indices(pipe.decomposition)
    # With an optimizer, labels are available for evaluation.
    dsl = generate_data(
        pipe;
        sampler=BoxSampler(-ones(3), ones(3)),
        n=4,
        rng=StableRNG(1),
        optimizer=HiGHS.Optimizer,
    )
    @test all(==(MOI.OPTIMAL), dsl.status)
    @test size(dsl.primal) == (8, 4)
    # complete / evaluate: the completion satisfies the equalities.
    rng = StableRNG(2)
    Z = randn(rng, 5, 10)
    pred = complete(pipe, ds.inputs, Z)
    @test pred isa Prediction && size(pred.primal) == (8, 10)
    @test max_equality_violation(prob, ds.inputs, pred) <= 1e-9
    rep = evaluate(pipe, ds.inputs, pred)
    @test maximum(rep.max_equality_violation) < 1e-9
    # Single instance (vector input).
    @test complete(pipe, ds.inputs[:, 1], Z[:, 1]).primal ≈ pred.primal[:, 1:1]
    # loss accepts the `proxy` keyword (training loops pass it).
    proxy = P -> zeros(5, size(P, 2))
    @test loss(pipe, ds.inputs, proxy(ds.inputs); proxy=proxy) isa Real
    # Decomposition-level entry points (used for GPU arrays).
    d = pipe.decomposition
    @test predict(pipe.method, d, ds.inputs, Z) ≈ pred.primal
    @test training_loss(pipe.method, d, ds.inputs, Z) ≈ loss(pipe, ds.inputs, Z)
    Yt = predict(pipe.method, d, ds.inputs, Z; training=true)
    @test training_loss(pipe.method, d, ds.inputs, Z) ≈
        soft_loss(d.form, Yt, ds.inputs; λ_ineq=10, λ_eq=10)
end

@testset "soft loss (eq. (2) of Donti et al.)" begin
    model, data = dc3_qp(; seed=3)
    prob = ParametricProblem(model)
    form = AffineForm(prob)
    rng = StableRNG(3)
    Y, P = randn(rng, 8, 4), uniform(rng, 3, 4)
    f = [0.5 * Y[:, k]' * data.Q * Y[:, k] + data.c' * Y[:, k] for k in 1:4]
    g = [sum(max.(data.G * Y[:, k] .- data.h, 0) .^ 2) for k in 1:4]
    h = [sum((data.A * Y[:, k] .- P[:, k]) .^ 2) for k in 1:4]
    @test soft_loss(form, Y, P; λ_ineq=2.0, λ_eq=3.0) ≈ sum(f .+ 2 .* g .+ 3 .* h) / 4
end

@testset "maximization problems" begin
    pmin = L2OPipeline(ParametricProblem(flow_lp()), DC3())
    pmax = L2OPipeline(ParametricProblem(flow_lp(; sense=MOI.MAX_SENSE)), DC3())
    rng = StableRNG(4)
    P, Z = uniform(rng, 2, 3; lo=0, hi=3), randn(rng, 2, 3)
    # max -f is min f: same loss, same predictions.
    @test loss(pmin, P, Z) ≈ loss(pmax, P, Z)
    @test complete(pmin, P, Z).primal ≈ complete(pmax, P, Z).primal
end

@testset "AD backends on the affine path" begin
    model, _ = dc3_qp(; seed=5)
    pipe = L2OPipeline(ParametricProblem(model), DC3(; train_steps=4))
    rng = StableRNG(5)
    P, Z = uniform(rng, 3, 3), randn(rng, 5, 3)
    gz = Zygote.gradient(z -> loss(pipe, P, z), Z)[1]
    gf = ForwardDiff.gradient(z -> loss(pipe, P, z), Z)
    @test gz ≈ gf
    pf = L2OPipeline(ParametricProblem(model), FSNet(; train_steps=4))
    Y = randn(rng, 8, 3)
    @test Zygote.gradient(y -> loss(pf, P, y), Y)[1] ≈
        ForwardDiff.gradient(y -> loss(pf, P, y), Y)
end

@testset "extension point: custom completion" begin
    # A third party can supply its own completion as a function
    # (problem, form) -> AbstractCompletion; here the built-in one with a fixed
    # dependent set.
    model, _ = dc3_qp(; seed=6)
    prob = ParametricProblem(model)
    custom = DC3(;
        completion=(problem, form) -> LinearCompletion(form; dependent=[1, 2, 3])
    )
    pipe = L2OPipeline(prob, custom)
    @test dependent_indices(pipe.decomposition) == [1, 2, 3]
    rng = StableRNG(6)
    P = uniform(rng, 3, 2)
    pred = complete(pipe, P, randn(rng, 5, 2))
    @test max_equality_violation(prob, P, pred) <= 1e-9
end

@testset "pipeline: FSNet" begin
    model, _ = dc3_qp(; seed=7)
    prob = ParametricProblem(model)
    pipe = L2OPipeline(prob, FSNet(; test_steps=2000, test_tol=1e-10))
    @test output_dimension(pipe) == 8
    ds = generate_data(pipe; sampler=BoxSampler(-ones(3), ones(3)), n=5, rng=StableRNG(7))
    @test ds.metadata["method"] == "FSNet"
    rng = StableRNG(8)
    pred = complete(pipe, ds.inputs, randn(rng, 8, 5))
    @test max_violation(prob, ds.inputs, pred) <= 1e-5
end
