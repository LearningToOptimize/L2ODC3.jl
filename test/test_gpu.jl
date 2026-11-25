# The affine path is written with dense matrix products and broadcasts only, so
# it runs on GPU arrays. JLArrays is a CPU-backed GPU array type that forbids
# scalar indexing, so this checks the device code path without a GPU.

@testset "DC3 on JLArray" begin
    model, _ = dc3_qp(; seed=31)
    method = DC3(; train_steps=3, test_steps=5)
    pipe = L2OPipeline(ParametricProblem(model), method)
    d = pipe.decomposition
    dj = Adapt.adapt(JLArray, d)
    @test dj.form.A isa JLArray
    @test dj.completion.Kz isa JLArray
    rng = StableRNG(31)
    P, Z = uniform(rng, 3, 4), randn(rng, 5, 4)
    Pj, Zj = JLArray(P), JLArray(Z)
    Yj = predict(method, dj, Pj, Zj)
    @test Yj isa JLArray
    @test Array(Yj) ≈ predict(method, d, P, Z)
    @test training_loss(method, dj, Pj, Zj) ≈ training_loss(method, d, P, Z)
    gj = Zygote.gradient(z -> training_loss(method, dj, Pj, z), Zj)[1]
    @test Array(gj) ≈ Zygote.gradient(z -> training_loss(method, d, P, z), Z)[1]
    # Single precision on the device.
    d32 = Adapt.adapt(JLArray{Float32}, d)
    @test eltype(d32.form.A) == Float32
    @test d32.step_size isa Float32
    @test eltype(d32.form.eqidx) == Int
    ℓ32 = training_loss(method, d32, JLArray{Float32}(P), JLArray{Float32}(Z))
    @test ℓ32 isa Float32
    @test ℓ32 ≈ training_loss(method, d, P, Z) rtol = 1e-4
end

@testset "FSNet on JLArray" begin
    model, _ = dc3_qp(; seed=32)
    method = FSNet(; train_steps=3, unroll=2, test_steps=5)
    pipe = L2OPipeline(ParametricProblem(model), method)
    d = pipe.decomposition
    dj = Adapt.adapt(JLArray, d)
    rng = StableRNG(32)
    P, Y = uniform(rng, 3, 4), randn(rng, 8, 4)
    Pj, Yj = JLArray(P), JLArray(Y)
    @test Array(predict(method, dj, Pj, Yj)) ≈ predict(method, d, P, Y)
    gj = Zygote.gradient(y -> training_loss(method, dj, Pj, y), Yj)[1]
    @test Array(gj) ≈ Zygote.gradient(y -> training_loss(method, d, P, y), Y)[1]
end
