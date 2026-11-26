# Equivalence with the authors' reference implementations, written out from
# their formulas (DC3: `SimpleProblem.complete_partial`, `ineq_partial_grad`,
# `grad_steps`, `total_loss`; FSNet: the training loss with distance weight
# ρ/2 and the early-stage penalty).

@testset "DC3 completion and correction step match the reference formulas" begin
    model, data = dc3_qp(; n=10, neq=4, nin=6, seed=41)
    prob = ParametricProblem(model)
    pipe = L2OPipeline(prob, DC3(; momentum=0.0))
    d = pipe.decomposition
    Zi, D = independent_indices(d), dependent_indices(d)
    A, G, h = data.A, data.G, data.h
    Ainv, Ap = inv(A[:, D]), A[:, Zi]
    rng = StableRNG(41)
    X, Z = uniform(rng, 4, 5), 2 .* randn(rng, 6, 5)
    # complete_partial: y_other = A_other⁻¹ (x - A_partial z).
    Yref = zeros(10, 5)
    Yref[Zi, :] = Z
    Yref[D, :] = Ainv * (X .- Ap * Z)
    Y = equality_completion(d.completion, Z, X)
    @test Y ≈ Yref
    # ineq_partial_grad: G_eff = G_partial - G_other A_other⁻¹ A_partial,
    # h_eff = h - G_other A_other⁻¹ x, grad = 2 G_effᵀ max(G_eff z - h_eff, 0),
    # Δy = [grad; -A_other⁻¹ A_partial grad].
    Geff = G[:, Zi] .- G[:, D] * Ainv * Ap
    heff = h .- G[:, D] * Ainv * X
    grad = 2 .* Geff' * max.(Geff * Z .- heff, 0)
    @test reduced_violation_gradient(d.form, d.completion, Y, X) ≈ grad
    ΔYref = zeros(10, 5)
    ΔYref[Zi, :] = grad
    ΔYref[D, :] = -(Ainv * Ap * grad)
    # grad_steps with momentum: s ← lr Δy + μ s, y ← y - s (in y-space), for
    # several steps; our z-step + re-completion gives the same iterates.
    for μ in (0.0, 0.5)
        lr = 0.3 * d.step_size
        Yr, S = copy(Yref), zero(Yref)
        for _ in 1:6
            gz = 2 .* Geff' * max.(Geff * Yr[Zi, :] .- heff, 0)
            Δ = zeros(10, 5)
            Δ[Zi, :] = gz
            Δ[D, :] = -(Ainv * Ap * gz)
            S = lr .* Δ .+ μ .* S
            Yr = Yr .- S
        end
        _, Yours = inequality_correction(
            d.form, d.completion, Z, X; steps=6, step_size=lr, momentum=μ
        )
        @test Yours ≈ Yr
    end
end

@testset "soft loss: squared (eq. (2)) and norm (reference code)" begin
    model, data = dc3_qp(; seed=42)
    prob = ParametricProblem(model)
    form = AffineForm(prob)
    rng = StableRNG(42)
    Y, X = randn(rng, 8, 4), uniform(rng, 3, 4)
    f = [0.5 * Y[:, k]' * data.Q * Y[:, k] + data.c' * Y[:, k] for k in 1:4]
    g = [norm(max.(data.G * Y[:, k] .- data.h, 0)) for k in 1:4]
    h = [norm(data.A * Y[:, k] .- X[:, k]) for k in 1:4]
    # total_loss of the reference code: softWeight = 10, softWeightEqFrac = 0.5.
    @test soft_loss(form, Y, X; λ_ineq=5, λ_eq=5, penalty=:norm) ≈
        sum(f .+ 5 .* g .+ 5 .* h) / 4
    @test soft_loss(form, Y, X; λ_ineq=5, λ_eq=5) ≈ sum(f .+ 5 .* g .^ 2 .+ 5 .* h .^ 2) / 4
    @test_throws ArgumentError soft_loss(form, Y, X; λ_ineq=1, λ_eq=1, penalty=:foo)
    @test_throws ArgumentError DC3(; penalty=:foo)
    # The smoothed norm has a finite gradient at feasible points.
    pipe = L2OPipeline(prob, DC3(; penalty=:norm, train_steps=2))
    Yf = pinv(data.A) * X
    Zf = Yf[independent_indices(pipe.decomposition), :]
    gz = Zygote.gradient(z -> loss(pipe, X, z), Zf)[1]
    @test all(isfinite, gz)
    @test grad_error(z -> loss(pipe, X, z), 3 .* randn(rng, 5, 4)) < 1e-7
end

@testset "FSNet loss matches the reference formula" begin
    model, data = dc3_qp(; seed=43)
    prob = ParametricProblem(model)
    m = FSNet(; train_steps=3)
    @test m.ρ == 10 && m.penalty == 10 && m.penalty_threshold == 1e3
    pipe = L2OPipeline(prob, m)
    d = pipe.decomposition
    rng = StableRNG(43)
    X = uniform(rng, 3, 4)
    for scale in (1.0, 30.0)   # below and above the early-penalty threshold Q
        Y = scale .* randn(rng, 8, 4)
        Ŷ = feasibility_seeking(d.form, Y, X; steps=3, step_size=d.step_size)
        φ = total_violation(d.form, Y, X)
        f = [0.5 * Ŷ[:, k]' * data.Q * Ŷ[:, k] + data.c' * Ŷ[:, k] for k in 1:4]
        ref = f .+ 5 .* vec(sum(abs2, Ŷ .- Y; dims=1)) .+ 10 .* φ .* (φ .>= 1e3)
        @test loss(pipe, X, Y) ≈ sum(ref) / 4
    end
end

@testset "heavy-ball correction with the default step converges (affine)" begin
    # Section 3.2 of Donti et al. states that for linear constraints the
    # correction converges; with momentum μ the default step
    # (1 - μ)/L is in the range where heavy ball provably converges.
    model, _ = dc3_qp(; n=10, neq=4, nin=6, seed=44)
    prob = ParametricProblem(model)
    for μ in (0.0, 0.5, 0.9)
        d = L2OPipeline(prob, DC3(; momentum=μ)).decomposition
        @test d.step_size ≈ (1 - μ) * default_step_size(d.form, d.completion)
        rng = StableRNG(44)
        X, Z = uniform(rng, 4, 6), 3 .* randn(rng, 6, 6)
        _, Y = inequality_correction(
            d.form, d.completion, Z, X; steps=4000, step_size=d.step_size, momentum=μ
        )
        @test maximum(inequality_violation(d.form, Y, X)) < 1e-10
    end
end
