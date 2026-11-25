# Named claims of DC3 and FSNet, each with a numerical check. The names are the
# anchors to which formal proofs can be attached later (see the Theory page).

"""
    CompletionJacobian()

Claim: the derivative used to backpropagate through the equality completion is
the true derivative of the completion map, i.e. for directions `u` (in `z`)
and `v` (in `y`), `vᵀ (∂y/∂z) u` computed with [`completion_pullback`](@ref)
(implicit function theorem, eqs. (3)–(4) of Donti et al.) equals the central
finite difference `vᵀ (φ(z + εu) - φ(z - εu)) / 2ε`.

Assumptions: `h` is C² and `J^h_D` is nonsingular at the completed points.

`check(CompletionJacobian(), pipe, inputs, output; atol = 1e-5, ε = 1e-6,
rng = Random.default_rng())` returns the relative error per instance.
"""
struct CompletionJacobian <: L2OCore.AbstractGuarantee end

"""
    CorrectionDescent()

Claim (descent lemma): if the reduced inequality violation
`v(z) = ‖ReLU(g_x([z; φ_x(z)]))‖²` has an `L`-Lipschitz gradient, every DC3
correction step without momentum and with `0 < γ ≤ 1/L` satisfies
`v(z_{k+1}) ≤ v(z_k) - (γ/2) ‖∇v(z_k)‖² ≤ v(z_k)`.

For affine constraints, `v(z) = dist²(Ĝz + ĥ(p), [ℓ, u])` with
`Ĝ = A_I K_z`, which is convex with `L = 2‖Ĝ‖₂²`, the default `1/L` step of
[`default_step_size`](@ref).

`check(CorrectionDescent(), pipe, inputs, output; steps = 20, atol = 1e-8)`
runs `steps` corrections with momentum `0` and the pipeline's step size, and
reports per instance the largest relative increase of `v` (and of the
sufficient-decrease bound).
"""
struct CorrectionDescent <: L2OCore.AbstractGuarantee end

"""
    FeasibilitySeekingDescent()

Claim (descent lemma, and Theorem 1 of Nguyen & Donti): if FSNet's violation
`φ(y) = ‖h(y)‖² + ‖ReLU(g(y))‖²` has an `L`-Lipschitz gradient and
`0 < η ≤ 1/L`, every gradient step satisfies
`φ(s_{k+1}) ≤ φ(s_k) - (η/2)‖∇φ(s_k)‖²`. For affine constraints, `φ` is convex
with `L = 2‖A‖₂²`.

`check(FeasibilitySeekingDescent(), pipe, inputs, output; steps = 20, atol = 1e-8)`
reports the largest relative violation of the sufficient-decrease inequality.
"""
struct FeasibilitySeekingDescent <: L2OCore.AbstractGuarantee end

function _result(g, errors::AbstractVector, atol::Real)
    e = collect(Float64, errors)
    worst = isempty(e) ? 0.0 : maximum(e)
    return L2OCore.CheckResult(g, worst <= atol, worst, e)
end

function L2OCore.check(
    g::CompletionJacobian,
    pipe::L2OCore.L2OPipeline,
    inputs,
    output;
    atol::Real=1e-5,
    ε::Real=1e-6,
    rng::Random.AbstractRNG=Random.default_rng(),
    kwargs...,
)
    d = pipe.decomposition
    d isa DC3Decomposition ||
        throw(ArgumentError("CompletionJacobian applies to DC3 pipelines"))
    P, Z = Matrix{Float64}(_asbatch(inputs)), Matrix{Float64}(_asbatch(output))
    c = d.completion
    Y = equality_completion(c, Z, P)
    U = randn(rng, size(Z)...)
    V = randn(rng, size(Y)...)
    Yp = equality_completion(c, Z .+ ε .* U, P)
    Ym = equality_completion(c, Z .- ε .* U, P)
    fd = vec(sum(V .* (Yp .- Ym); dims=1)) ./ (2ε)
    an = vec(sum(U .* completion_pullback(c, Y, P, V); dims=1))
    err = abs.(fd .- an) ./ max.(1, abs.(an))
    return _result(g, err, atol)
end

function L2OCore.check(
    g::CorrectionDescent,
    pipe::L2OCore.L2OPipeline,
    inputs,
    output;
    steps::Integer=20,
    atol::Real=1e-8,
    kwargs...,
)
    d = pipe.decomposition
    d isa DC3Decomposition ||
        throw(ArgumentError("CorrectionDescent applies to DC3 pipelines"))
    P, Z = _asbatch(inputs), _asbatch(output)
    γ = d.step_size
    Y = equality_completion(d.completion, Z, P)
    err = zeros(size(Z, 2))
    for _ in 1:steps
        v = inequality_violation(d.form, Y, P)
        G = reduced_violation_gradient(d.form, d.completion, Y, P)
        Z = Z .- γ .* G
        Y = equality_completion(d.completion, Z, P; init=Y)
        vnew = inequality_violation(d.form, Y, P)
        bound = v .- (γ / 2) .* _colsumsq(G)
        err .= max.(err, max.(vnew .- bound, 0) ./ max.(1, v))
    end
    return _result(g, err, atol)
end

function L2OCore.check(
    g::FeasibilitySeekingDescent,
    pipe::L2OCore.L2OPipeline,
    inputs,
    output;
    steps::Integer=20,
    atol::Real=1e-8,
    kwargs...,
)
    d = pipe.decomposition
    d isa FSNetDecomposition ||
        throw(ArgumentError("FeasibilitySeekingDescent applies to FSNet pipelines"))
    P, S = _asbatch(inputs), _asbatch(output)
    η = d.step_size
    err = zeros(size(S, 2))
    for _ in 1:steps
        φ = total_violation(d.form, S, P)
        G = violation_gradient(d.form, S, P)
        S = S .- η .* G
        φnew = total_violation(d.form, S, P)
        bound = φ .- (η / 2) .* _colsumsq(G)
        err .= max.(err, max.(φnew .- bound, 0) ./ max.(1, φ))
    end
    return _result(g, err, atol)
end
