# Inequality correction (Section 3.2 of Donti, Rolnick & Kolter, 2021) and
# feasibility seeking (eq. (4) of Nguyen & Donti, 2025).

"""
    reduced_violation_gradient(form, completion, Y, P) -> ΔZ

Gradient of the inequality violation with respect to the independent
variables, at completed points `Y = [z; φ_x(z)]`:

```math
\\Delta z = \\nabla_z \\big\\|\\mathrm{ReLU}\\big(g_x([z;\\varphi_x(z)])\\big)\\big\\|_2^2
         = \\Big(\\frac{\\partial y}{\\partial z}\\Big)^{\\!\\top} J^g(y)^\\top\\, 2\\,\\mathrm{ReLU}(g_x(y)),
```

the correction direction of DC3 (Section 3.2 of Donti et al.). Computed with
[`constraint_vjp`](@ref) and [`completion_pullback`](@ref), so it is itself
differentiable (needed to backpropagate through unrolled corrections).
"""
function reduced_violation_gradient(
    form::AbstractProblemForm, c::AbstractCompletion, Y::AbstractMatrix, P::AbstractMatrix
)
    W = _inequality_weights(form, Y, P)
    Gy = constraint_vjp(form, Y, P, W)
    return completion_pullback(c, Y, P, Gy)
end

# 2·(F - Π(F)) on inequality rows, zero elsewhere (m × B).
function _inequality_weights(
    form::AbstractProblemForm, Y::AbstractMatrix, P::AbstractMatrix
)
    F = L2OCore.constraint_values(form, Y, P)
    return 2 .* _nograd(() -> _inmask(form)) .* projection_residual(form, F)
end

# 2·(F - Π(F)) on all box rows: ∇_F φ for FSNet's φ = ‖h‖² + ‖g⁺‖².
function _violation_weights(form::AbstractProblemForm, Y::AbstractMatrix, P::AbstractMatrix)
    F = L2OCore.constraint_values(form, Y, P)
    return 2 .* projection_residual(form, F)
end

function _inmask(form::AbstractProblemForm)
    lower, _ = row_bounds(form)
    mask = similar(lower)
    fill!(mask, zero(eltype(mask)))
    mask[violation_indices(form)] .= one(eltype(mask))
    return mask
end

"""
    inequality_correction(form, completion, Z, P; steps, step_size, momentum = 0,
                          tol = 0) -> (Z, Y)

DC3's correction `ρ_x^{(t)}`: `steps` gradient steps on the inequality
violation in the independent variables, with heavy-ball momentum as in the
reference implementation,

```math
v_{k+1} = \\mu v_k + \\Delta z_k,\\qquad z_{k+1} = z_k - \\gamma\\, v_{k+1},\\qquad
y_{k+1} = [z_{k+1};\\varphi_x(z_{k+1})],
```

where `Δz_k` is [`reduced_violation_gradient`](@ref). Returns the corrected
independent variables and the completed points.

Each iterate is re-completed. For affine equalities this is identical to the
paper's step `y - γ[Δz; (∂φ/∂z) Δz]`; for nonlinear equalities the paper's
step keeps the equalities only to first order, while re-completion keeps them
to the Newton tolerance (a deliberate deviation).

`tol > 0` stops early once every instance has `max ReLU(g) ≤ tol` (use at test
time only; the stopping test is not differentiated).

If the reduced violation `v(z) = ‖ReLU(g_x([z; φ_x(z)]))‖²` has an
`L`-Lipschitz gradient, `momentum = 0` and `0 < step_size ≤ 1/L`, then
`v(z_{k+1}) ≤ v(z_k) - (γ/2)‖Δz_k‖²` (descent lemma; see the Theory page).
"""
function inequality_correction(
    form::AbstractProblemForm,
    c::AbstractCompletion,
    Z::AbstractMatrix,
    P::AbstractMatrix;
    steps::Integer,
    step_size::Real,
    momentum::Real=0,
    tol::Real=0,
)
    Y = equality_completion(c, Z, P)
    V = zero(Z)
    for _ in 1:steps
        if tol > 0 && _nograd(() -> _max_inequality_violation(form, Y, P)) <= tol
            break
        end
        G = reduced_violation_gradient(form, c, Y, P)
        V = momentum .* V .+ G
        Z = Z .- step_size .* V
        Y = equality_completion(c, Z, P; init=_stopgrad(Y))
    end
    return Z, Y
end

function _max_inequality_violation(form, Y, P)
    return maximum(abs, inequality_residual(form, Y, P); init=zero(eltype(Y)))
end

"""
    violation_gradient(form, Y, P) -> AbstractMatrix

`∇_y φ(y)` for FSNet's violation `φ(y) = ‖h_x(y)‖² + ‖ReLU(g_x(y))‖²`
(eq. (3) of Nguyen & Donti), i.e. `J_yᵀ 2(F - Π(F))`.
"""
violation_gradient(form::AbstractProblemForm, Y::AbstractMatrix, P::AbstractMatrix) =
    constraint_vjp(form, Y, P, _violation_weights(form, Y, P))

"""
    feasibility_seeking(form, Y, P; steps, step_size, unroll = steps, tol = 0) -> Ŷ

FSNet's feasibility-seeking map `FS` truncated to `steps` gradient-descent
iterations on `φ(s) = ‖h_x(s)‖² + ‖ReLU(g_x(s))‖²` started at the proxy output
(eq. (4) of Nguyen & Donti, 2025):

```math
s_0 = y_\\theta(x),\\qquad s_{k+1} = s_k - \\eta_\\phi \\nabla_s \\phi(s_k; x).
```

Derivatives flow through the first `unroll` iterations only (truncated
unrolled differentiation, Section 5 of the paper); the remaining iterations
contribute a constant offset. `tol > 0` stops early once `φ ≤ tol` for every
instance (not differentiated).

With `φ` `L`-smooth and `0 < η ≤ 1/L`, `φ(s_{k+1}) ≤ φ(s_k)`; if moreover
`φ` satisfies the Polyak–Łojasiewicz inequality with constant `μ` and a
feasible point exists, `φ(s_k) ≤ (1 - μ/L)^k φ(s_0)` (Theorem 1 of the paper). For affine
constraints with a nonempty feasible set both assumptions hold (Hoffman's
error bound), with `L` from [`lipschitz_constant`](@ref).
"""
function feasibility_seeking(
    form::AbstractProblemForm,
    Y::AbstractMatrix,
    P::AbstractMatrix;
    steps::Integer,
    step_size::Real,
    unroll::Integer=steps,
    tol::Real=0,
)
    0 <= unroll || throw(ArgumentError("unroll must be nonnegative"))
    S = Y
    K = min(unroll, steps)
    for _ in 1:K
        if tol > 0 && _nograd(() -> _max_total_violation(form, S, P)) <= tol
            return S
        end
        S = S .- step_size .* violation_gradient(form, S, P)
    end
    K == steps && return S
    T = _stopgrad(_continue_descent(form, S, P, steps - K, step_size, tol))
    return S .+ (T .- _stopgrad(S))
end

function _continue_descent(form, S, P, steps, step_size, tol)
    for _ in 1:steps
        if tol > 0 && _nograd(() -> _max_total_violation(form, S, P)) <= tol
            break
        end
        S = S .- step_size .* violation_gradient(form, S, P)
    end
    return S
end

function _max_total_violation(form, Y, P)
    return maximum(total_violation(form, Y, P); init=zero(eltype(Y)))
end
