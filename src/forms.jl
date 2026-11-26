# Problem forms: differentiable, batched access to f, h and g of problem (1) of
# Donti, Rolnick & Kolter (2021),
#
#     min_y f_x(y)   s.t.   g_x(y) ≤ 0,   h_x(y) = 0,
#
# where the parameters (the paper's `x`) are the JuMP parameters `P` and the
# decision variables `y` are the columns of `Y`. Every constraint row is stored
# as `lower ≤ F(y, p) ≤ upper` in `L2OCore.dual_layout` order: rows with
# `lower == upper` are equalities `h`, the others are inequalities `g`.

"""
    AbstractProblemForm <: L2OCore.AbstractEvaluator

Differentiable batch representation of a parametric problem used by DC3 and
FSNet. A form is an `L2OCore.AbstractEvaluator` (it implements
`batch_objective`, `constraint_values`, `row_violation` and `equality_rows`),
and additionally provides

| function | returns |
|:--|:--|
| `L2OCore.num_decision_variables(form)` | number of decision variables `n` (rows of `Y`) |
| [`row_bounds`](@ref)`(form)` | `(lower, upper)`, one entry per row |
| [`equality_indices`](@ref)`(form)` / [`inequality_indices`](@ref)`(form)` | row indices of `h` and `g` |
| [`objective_sign`](@ref)`(form)` | `+1` (min) or `-1` (max) |
| [`constraint_vjp`](@ref)`(form, Y, P, W)` | `J_y F(y_k, p_k)ᵀ W[:, k]` for every column |

All functions are batch-first (batch = last dimension) and differentiable with
ChainRules-compatible reverse-mode AD (e.g. Zygote), including
`constraint_vjp`, whose derivative needs second derivatives of `F`.

Built-in forms: [`AffineForm`](@ref) (dense arrays, GPU-ready) and
[`NLPForm`](@ref) (JuMP nonlinear models through `MOI.Nonlinear`). A new
backend (for instance batched GPU kernels generated from ExaModels) subtypes
`AbstractProblemForm` and implements the functions above.
"""
abstract type AbstractProblemForm <: L2OCore.AbstractEvaluator end

"""
    row_bounds(form::AbstractProblemForm) -> (lower, upper)

Bounds of every constraint row: row `i` is `lower[i] ≤ F_i(y, p) ≤ upper[i]`.
"""
function row_bounds end

"""
    equality_indices(form::AbstractProblemForm) -> AbstractVector{Int}

Rows with `lower == upper`, the equality constraints `h_x(y) = 0` of DC3.
"""
function equality_indices end

"""
    inequality_indices(form::AbstractProblemForm) -> AbstractVector{Int}

Rows with `lower < upper` and at least one finite bound, the inequality
constraints `g_x(y) ≤ 0` of DC3 (two-sided rows count once).
"""
function inequality_indices end

"""
    objective_sign(form::AbstractProblemForm) -> Real

`+1` for minimization (and feasibility) problems, `-1` for maximization: a
method of `L2OCore.objective_sign`, so that `objective_sign(form) * f` is always
minimized.
"""
L2OCore.objective_sign(::AbstractProblemForm)

"""
    constraint_vjp(form::AbstractProblemForm, Y, P, W) -> AbstractMatrix

Vector-Jacobian product of the constraint function for every instance:
column `k` is `J_y F(Y[:, k], P[:, k])ᵀ W[:, k]` (`n × B`), where
`W` has one row per constraint row. Differentiable in `Y`, `P` and `W`.
"""
function constraint_vjp end

# Shared batch helpers ---------------------------------------------------------

# Treat `x` as a constant for AD.
_stopgrad(x) = ChainRulesCore.ignore_derivatives(x)
# Run `f()` outside of AD (avoids generating rules, and their keyword methods).
_nograd(f) = ChainRulesCore.ignore_derivatives(f)

"""
    min_objective(form, Y, P) -> AbstractVector

Objective of every column in minimization form, `objective_sign(form) * f`.
"""
min_objective(form::AbstractProblemForm, Y::AbstractMatrix, P::AbstractMatrix) =
    objective_sign(form) .* L2OCore.batch_objective(form, Y, P)

# Signed distance of each row to its interval: `F - clamp(F, lower, upper)`.
# Its square is `ReLU(g)^2` for inequality rows and `h^2` for equality rows, and
# `∇_F ½‖F - Π(F)‖² = F - Π(F)` (the projection residual).
_projection_residual(F, lower, upper) = F .- clamp.(F, lower, upper)

"""
    equality_residual(form, Y, P) -> AbstractMatrix

`h_x(y)` for every column (`num_equalities × B`).
"""
function equality_residual(form::AbstractProblemForm, Y::AbstractMatrix, P::AbstractMatrix)
    F = L2OCore.constraint_values(form, Y, P)
    lower, _ = row_bounds(form)
    eq = equality_indices(form)
    return F[eq, :] .- lower[eq]
end

"""
    inequality_residual(form, Y, P) -> AbstractMatrix

`ReLU(g_x(y))` for every column, as the signed distance of each inequality row
to its interval (`num_inequalities × B`); its squared norm is the inequality
violation term of the DC3 soft loss (eq. (2) of Donti et al.).
"""
function inequality_residual(
    form::AbstractProblemForm, Y::AbstractMatrix, P::AbstractMatrix
)
    F = L2OCore.constraint_values(form, Y, P)
    lower, upper = row_bounds(form)
    in = inequality_indices(form)
    return _projection_residual(F[in, :], lower[in], upper[in])
end

"""
    inequality_violation(form, Y, P) -> AbstractVector

`‖ReLU(g_x(y))‖²₂` for every column.
"""
inequality_violation(form::AbstractProblemForm, Y::AbstractMatrix, P::AbstractMatrix) =
    _colsumsq(inequality_residual(form, Y, P))

"""
    equality_violation(form, Y, P) -> AbstractVector

`‖h_x(y)‖²₂` for every column.
"""
equality_violation(form::AbstractProblemForm, Y::AbstractMatrix, P::AbstractMatrix) =
    _colsumsq(equality_residual(form, Y, P))

"""
    total_violation(form, Y, P) -> AbstractVector

FSNet's violation function `φ(y) = ‖h_x(y)‖²₂ + ‖ReLU(g_x(y))‖²₂` (eq. (3) of
Nguyen & Donti, 2025) for every column.
"""
function total_violation(form::AbstractProblemForm, Y::AbstractMatrix, P::AbstractMatrix)
    F = L2OCore.constraint_values(form, Y, P)
    lower, upper = row_bounds(form)
    return _colsumsq(_projection_residual(F, lower, upper))
end

_colsumsq(R::AbstractMatrix) = vec(sum(abs2, R; dims=1))

function L2OCore.row_violation(
    form::AbstractProblemForm, Y::AbstractMatrix, P::AbstractMatrix
)
    return abs.(
        _projection_residual(L2OCore.constraint_values(form, Y, P), row_bounds(form)...)
    )
end

# Rows that are boxes with at least one finite side and are not equalities.
function _inequality_mask(lower::AbstractVector, upper::AbstractVector, eq::BitVector)
    return .!eq .& (isfinite.(lower) .| isfinite.(upper))
end

# =============================================================================
# AffineForm
# =============================================================================

"""
    AffineForm(problem::L2OCore.ParametricProblem; T = Float64)
    AffineForm(ev::L2OCore.MatrixEvaluator)

Dense matrix form of a problem with affine constraints and an (at most)
quadratic objective, parameters entering affinely:

```math
\\min_y \\; \\tfrac12 y^\\top Q y + c^\\top y + y^\\top H p + c_p^\\top p + \\tfrac12 p^\\top Q_{pp} p + c_0
\\quad\\text{s.t.}\\quad \\ell \\le A y + B p + b \\le u .
```

Every operation is a dense matrix product or a broadcast, so the form runs
unchanged on GPU arrays once its fields are moved with `Adapt.adapt` (see the
package extension) and is differentiable by any array AD.

Assumptions: every constraint is a box-like set (`EqualTo`, `GreaterThan`,
`LessThan`, `Interval`, `Zeros`, `Nonnegatives`, `Nonpositives`); problems with
second-order or other non-decomposable cones are rejected.
"""
struct AffineForm{
    T<:Real,M<:AbstractMatrix{T},V<:AbstractVector{T},I<:AbstractVector{Int}
} <: AbstractProblemForm
    sense::T
    A::M
    B::M
    b::V
    lower::V
    upper::V
    Q::M
    c::V
    H::M
    cp::V
    Qpp::M
    c0::T
    eq::BitVector
    bnd::BitVector
    eqidx::I
    inidx::I
end

function AffineForm(ev::L2OCore.MatrixEvaluator{T}) where {T}
    isempty(ev.cones) || throw(
        ArgumentError(
            "AffineForm supports only box-like constraint sets; the problem has " *
            "$(length(ev.cones)) non-decomposable cone(s) " *
            "($(join(unique(nameof(typeof(s)) for (_, s) in ev.cones), ", "))). " *
            "DC3/FSNet need a differentiable violation of every inequality.",
        ),
    )
    eq = BitVector(ev.eq)
    inmask = _inequality_mask(ev.lower, ev.upper, eq)
    D(M) = Matrix{T}(M)
    return AffineForm(
        T(ev.sense),
        D(ev.A),
        D(ev.B),
        Vector{T}(ev.b),
        Vector{T}(ev.lower),
        Vector{T}(ev.upper),
        D(ev.Q),
        Vector{T}(ev.c),
        D(ev.H),
        Vector{T}(ev.cp),
        D(ev.Qpp),
        T(ev.c0),
        eq,
        BitVector(ev.bnd),
        findall(eq),
        findall(inmask),
    )
end

function AffineForm(problem::L2OCore.ParametricProblem; T::Type{<:Real}=Float64)
    return AffineForm(L2OCore.MatrixEvaluator(problem; T=T))
end

L2OCore.num_decision_variables(f::AffineForm) = size(f.A, 2)
row_bounds(f::AffineForm) = (f.lower, f.upper)
equality_indices(f::AffineForm) = f.eqidx
inequality_indices(f::AffineForm) = f.inidx
L2OCore.objective_sign(f::AffineForm) = f.sense
L2OCore.equality_rows(f::AffineForm) = f.eq
L2OCore.bound_rows(f::AffineForm) = f.bnd

_quadform(Q, X) = vec(sum(X .* (Q * X); dims=1))

function L2OCore.batch_objective(f::AffineForm, Y::AbstractMatrix, P::AbstractMatrix)
    return vec(f.c' * Y) .+ vec(f.cp' * P) .+ f.c0 .+ _quadform(f.Q, Y) ./ 2 .+
           vec(sum(Y .* (f.H * P); dims=1)) .+ _quadform(f.Qpp, P) ./ 2
end

function L2OCore.constraint_values(f::AffineForm, Y::AbstractMatrix, P::AbstractMatrix)
    return f.A * Y .+ f.B * P .+ f.b
end

function constraint_vjp(
    f::AffineForm, Y::AbstractMatrix, P::AbstractMatrix, W::AbstractMatrix
)
    return f.A' * W
end

"""
    lipschitz_constant(form::AffineForm; rows = :all) -> Real

Lipschitz constant `L = 2‖A_R‖₂²` of the gradient of `‖F_R - Π(F_R)‖²` on the
rows `R` (`:all` box rows, `:inequality` or `:equality`). Gradient descent with
step `1/L` on that function never increases it (descent lemma); see the Theory
page.
"""
function lipschitz_constant(f::AffineForm; rows::Symbol=:all)
    R = if rows === :all
        sort!(vcat(collect(f.eqidx), collect(f.inidx)))
    elseif rows === :inequality
        collect(f.inidx)
    elseif rows === :equality
        collect(f.eqidx)
    else
        throw(ArgumentError("rows must be :all, :inequality or :equality"))
    end
    isempty(R) && return zero(eltype(f.A))
    return 2 * LinearAlgebra.opnorm(Matrix(f.A[R, :]))^2
end

# =============================================================================
# NLPForm (JuMP nonlinear models through MOI.Nonlinear)
# =============================================================================

"""
    NLPForm(problem::L2OCore.ParametricProblem)

Form of a general smooth problem (nonlinear, quadratic or affine functions)
built from the JuMP model with `MathOptInterface.Nonlinear` reverse-mode AD.
Parameters are treated as extra evaluator variables, so derivatives with
respect to the parameters come for free.

Batches are evaluated column by column on the CPU. Every function has a
ChainRules `rrule`: `batch_objective` and `constraint_values` use
gradients/Jacobians; `constraint_vjp` uses Hessian-of-the-Lagrangian products,
so unrolled DC3 corrections are differentiated exactly.

Assumptions: twice continuously differentiable functions; box-like sets only
(as for [`AffineForm`](@ref)); no legacy `@NL` macros. Not thread-safe (the
form owns evaluation buffers).
"""
struct NLPForm{E<:MOI.Nonlinear.Evaluator} <: AbstractProblemForm
    evaluator::E
    n::Int
    np::Int
    m::Int
    sense::Float64
    lower::Vector{Float64}
    upper::Vector{Float64}
    eq::BitVector
    bnd::BitVector
    affine::BitVector
    eqidx::Vector{Int}
    inidx::Vector{Int}
    jac_structure::Vector{Tuple{Int,Int}}
    jac_values::Vector{Float64}
    xbuf::Vector{Float64}
    gbuf::Vector{Float64}
    hbuf::Vector{Float64}
end

const _BOX_SETS = Union{
    MOI.EqualTo,
    MOI.GreaterThan,
    MOI.LessThan,
    MOI.Interval,
    MOI.Zeros,
    MOI.Nonnegatives,
    MOI.Nonpositives,
}

_bounds(s::MOI.EqualTo) = (s.value, s.value)
_bounds(s::MOI.GreaterThan) = (s.lower, Inf)
_bounds(s::MOI.LessThan) = (-Inf, s.upper)
_bounds(s::MOI.Interval) = (s.lower, s.upper)
_bounds(::MOI.Zeros) = (0.0, 0.0)
_bounds(::MOI.Nonnegatives) = (0.0, Inf)
_bounds(::MOI.Nonpositives) = (-Inf, 0.0)

_is_affine(::Union{JuMP.AbstractVariableRef,JuMP.GenericAffExpr}) = true
_is_affine(f::AbstractVector) = all(_is_affine, f)
_is_affine(::Any) = false

_snf(f::MOI.ScalarNonlinearFunction) = f
_snf(f) = convert(MOI.ScalarNonlinearFunction, f)
_scalar_rows(f::MOI.AbstractScalarFunction) = [f]
_scalar_rows(f::MOI.AbstractVectorFunction) = collect(MOI.Utilities.eachscalar(f))

function NLPForm(problem::L2OCore.ParametricProblem)
    model = problem.model
    JuMP.nonlinear_model(model) === nothing || throw(
        ArgumentError(
            "NLPForm does not support legacy `@NL` macros; use `@constraint`/`@objective`",
        ),
    )
    vars = L2OCore.decision_variables(problem)
    params = L2OCore.parameters(problem)
    n, np, m = length(vars), length(params), L2OCore.num_duals(problem)
    nlp = MOI.Nonlinear.Model()
    obj = JuMP.moi_function(JuMP.objective_function(model))
    MOI.Nonlinear.set_objective(nlp, _snf(obj))
    lower, upper = fill(-Inf, m), fill(Inf, m)
    bnd = falses(m)
    affine = falses(m)
    for (c, rows) in zip(L2OCore.constraint_refs(problem), L2OCore.dual_layout(problem))
        cobj = JuMP.constraint_object(c)
        set = JuMP.moi_set(cobj)
        set isa _BOX_SETS || throw(
            ArgumentError(
                "constraint set $(nameof(typeof(set))) is not supported; DC3/FSNet need " *
                "box-like sets (EqualTo, GreaterThan, LessThan, Interval, Zeros, " *
                "Nonnegatives, Nonpositives)",
            ),
        )
        l, u = _bounds(set)
        fs = _scalar_rows(JuMP.moi_function(cobj.func))
        length(fs) == length(rows) || error("internal error: row count mismatch")
        for (r, fr) in zip(rows, fs)
            # The set given to MOI.Nonlinear is irrelevant: we only evaluate F.
            MOI.Nonlinear.add_constraint(nlp, _snf(fr), MOI.EqualTo(0.0))
            lower[r], upper[r] = l, u
        end
        bnd[rows] .=
            cobj.func isa JuMP.AbstractVariableRef ||
            cobj.func isa AbstractVector{<:JuMP.AbstractVariableRef}
        affine[rows] .= _is_affine(cobj.func)
    end
    order = vcat(JuMP.index.(vars), JuMP.index.(params))
    ev = MOI.Nonlinear.Evaluator(nlp, MOI.Nonlinear.SparseReverseMode(), order)
    MOI.initialize(ev, [:Grad, :Jac, :HessVec])
    js = MOI.jacobian_structure(ev)
    eq = lower .== upper
    inmask = _inequality_mask(lower, upper, eq)
    s = JuMP.objective_sense(model) == MOI.MAX_SENSE ? -1.0 : 1.0
    return NLPForm(
        ev,
        n,
        np,
        m,
        s,
        lower,
        upper,
        eq,
        bnd,
        affine,
        findall(eq),
        findall(inmask),
        js,
        zeros(length(js)),
        zeros(n + np),
        zeros(m),
        zeros(n + np),
    )
end

L2OCore.num_decision_variables(f::NLPForm) = f.n
row_bounds(f::NLPForm) = (f.lower, f.upper)
equality_indices(f::NLPForm) = f.eqidx
inequality_indices(f::NLPForm) = f.inidx
L2OCore.objective_sign(f::NLPForm) = f.sense
L2OCore.equality_rows(f::NLPForm) = f.eq
L2OCore.bound_rows(f::NLPForm) = f.bnd

function _setx!(f::NLPForm, y::AbstractVector, p::AbstractVector)
    copyto!(view(f.xbuf, 1:(f.n)), y)
    copyto!(view(f.xbuf, (f.n + 1):(f.n + f.np)), p)
    return f.xbuf
end

function _check_batch(f::NLPForm, Y::AbstractMatrix, P::AbstractMatrix)
    size(Y, 1) == f.n ||
        throw(DimensionMismatch("Y has $(size(Y, 1)) rows, expected $(f.n)"))
    size(P, 1) == f.np ||
        throw(DimensionMismatch("P has $(size(P, 1)) rows, expected $(f.np)"))
    size(Y, 2) == size(P, 2) ||
        throw(DimensionMismatch("Y and P have different batch sizes"))
    return nothing
end

"""
    constraint_jacobian(form::NLPForm, y, p) -> Matrix

Dense Jacobian `[J_y F  J_p F]` (`m × (n + num_parameters)`) at one instance.
"""
function constraint_jacobian(f::NLPForm, y::AbstractVector, p::AbstractVector)
    x = _setx!(f, y, p)
    MOI.eval_constraint_jacobian(f.evaluator, f.jac_values, x)
    J = zeros(f.m, f.n + f.np)
    for (k, (i, j)) in enumerate(f.jac_structure)
        J[i, j] += f.jac_values[k]
    end
    return J
end

function _constraint_values(f::NLPForm, Y::AbstractMatrix, P::AbstractMatrix)
    _check_batch(f, Y, P)
    F = zeros(f.m, size(Y, 2))
    for k in axes(Y, 2)
        x = _setx!(f, view(Y, :, k), view(P, :, k))
        MOI.eval_constraint(f.evaluator, f.gbuf, x)
        F[:, k] .= f.gbuf
    end
    return F
end

function L2OCore.constraint_values(f::NLPForm, Y::AbstractMatrix, P::AbstractMatrix)
    return _constraint_values(f, Y, P)
end

function ChainRulesCore.rrule(
    ::typeof(L2OCore.constraint_values), f::NLPForm, Y::AbstractMatrix, P::AbstractMatrix
)
    F = _constraint_values(f, Y, P)
    function constraint_values_pullback(ΔF)
        F̄ = ChainRulesCore.unthunk(ΔF)
        Ȳ, P̄ = zeros(size(Y)), zeros(size(P))
        for k in axes(Y, 2)
            J = constraint_jacobian(f, view(Y, :, k), view(P, :, k))
            g = J' * view(F̄, :, k)
            Ȳ[:, k] .= g[1:(f.n)]
            P̄[:, k] .= g[(f.n + 1):end]
        end
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), Ȳ, P̄
    end
    return F, constraint_values_pullback
end

function _objective(f::NLPForm, Y::AbstractMatrix, P::AbstractMatrix)
    _check_batch(f, Y, P)
    return [
        MOI.eval_objective(f.evaluator, _setx!(f, view(Y, :, k), view(P, :, k))) for
        k in axes(Y, 2)
    ]
end

function L2OCore.batch_objective(f::NLPForm, Y::AbstractMatrix, P::AbstractMatrix)
    return _objective(f, Y, P)
end

function ChainRulesCore.rrule(
    ::typeof(L2OCore.batch_objective), f::NLPForm, Y::AbstractMatrix, P::AbstractMatrix
)
    v = _objective(f, Y, P)
    function batch_objective_pullback(Δv)
        v̄ = ChainRulesCore.unthunk(Δv)
        Ȳ, P̄ = zeros(size(Y)), zeros(size(P))
        g = zeros(f.n + f.np)
        for k in axes(Y, 2)
            x = _setx!(f, view(Y, :, k), view(P, :, k))
            MOI.eval_objective_gradient(f.evaluator, g, x)
            Ȳ[:, k] .= v̄[k] .* g[1:(f.n)]
            P̄[:, k] .= v̄[k] .* g[(f.n + 1):end]
        end
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), Ȳ, P̄
    end
    return v, batch_objective_pullback
end

function _constraint_vjp(
    f::NLPForm, Y::AbstractMatrix, P::AbstractMatrix, W::AbstractMatrix
)
    _check_batch(f, Y, P)
    size(W) == (f.m, size(Y, 2)) || throw(DimensionMismatch("W must be $(f.m) × B"))
    G = zeros(f.n, size(Y, 2))
    for k in axes(Y, 2)
        J = constraint_jacobian(f, view(Y, :, k), view(P, :, k))
        G[:, k] .= view(J, :, 1:(f.n))' * view(W, :, k)
    end
    return G
end

function constraint_vjp(f::NLPForm, Y::AbstractMatrix, P::AbstractMatrix, W::AbstractMatrix)
    return _constraint_vjp(f, Y, P, W)
end

# Hessian of the weighted constraint sum `Σᵢ μᵢ ∇²Fᵢ(x)` times `v` (x = [y; p]).
function _hessian_product!(out, f::NLPForm, y, p, v::AbstractVector, μ::AbstractVector)
    x = _setx!(f, y, p)
    MOI.eval_hessian_lagrangian_product(f.evaluator, out, x, v, 0.0, μ)
    return out
end

function ChainRulesCore.rrule(
    ::typeof(constraint_vjp),
    f::NLPForm,
    Y::AbstractMatrix,
    P::AbstractMatrix,
    W::AbstractMatrix,
)
    G = _constraint_vjp(f, Y, P, W)
    function constraint_vjp_pullback(ΔG)
        Ḡ = ChainRulesCore.unthunk(ΔG)
        Ȳ, P̄, W̄ = zeros(size(Y)), zeros(size(P)), zeros(size(W))
        v = zeros(f.n + f.np)
        for k in axes(Y, 2)
            y, p = view(Y, :, k), view(P, :, k)
            J = constraint_jacobian(f, y, p)
            # G_k = J_y(x)ᵀ w  ⇒  ∂/∂w = J_y Ḡ_k,  ∂/∂x = (Σᵢ wᵢ ∇²Fᵢ(x)) [Ḡ_k; 0].
            W̄[:, k] .= view(J, :, 1:(f.n)) * view(Ḡ, :, k)
            v[1:(f.n)] .= view(Ḡ, :, k)
            v[(f.n + 1):end] .= 0
            _hessian_product!(f.hbuf, f, y, p, v, collect(view(W, :, k)))
            Ȳ[:, k] .= view(f.hbuf, 1:(f.n))
            P̄[:, k] .= view(f.hbuf, (f.n + 1):(f.n + f.np))
        end
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), Ȳ, P̄, W̄
    end
    return G, constraint_vjp_pullback
end

# =============================================================================
# Form selection
# =============================================================================

"""
    problem_form(problem::L2OCore.ParametricProblem; T = Float64) -> AbstractProblemForm

[`AffineForm`](@ref) when the problem has a matrix form (affine constraints,
at most quadratic objective, parameters entering affinely), otherwise
[`NLPForm`](@ref).
"""
function problem_form(problem::L2OCore.ParametricProblem; T::Type{<:Real}=Float64)
    if L2OCore.has_matrix_form(problem)
        return AffineForm(problem; T=T)
    end
    return NLPForm(problem)
end
