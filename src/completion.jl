# Equality completion (Section 3.1 of Donti, Rolnick & Kolter, 2021).
#
# The decision vector is split into independent variables `z` (predicted by
# the proxy) and dependent variables `φ_x(z)` defined implicitly by
# `h_x([z; φ_x(z)]) = 0`. Derivatives follow the implicit function theorem,
# eqs. (3)–(4) of the paper.

"""
    AbstractCompletion

Equality completion `z ↦ y = [z; φ_x(z)]` of DC3. A completion implements

| function | returns |
|:--|:--|
| [`independent_indices`](@ref)`(c)` | positions of `z` inside `y` |
| [`dependent_indices`](@ref)`(c)` | positions of `φ_x(z)` inside `y` |
| [`equality_completion`](@ref)`(c, Z, P)` | `Y` with `h(Y, P) = 0` (differentiable in `Z`, `P`) |
| [`completion_pullback`](@ref)`(c, Y, P, Ȳ)` | `(∂y/∂z)ᵀ Ȳ` (differentiable in `Y`, `P`, `Ȳ`) |

Built-in: [`LinearCompletion`](@ref) (affine equalities, closed form) and
[`NewtonCompletion`](@ref) (nonlinear equalities).
"""
abstract type AbstractCompletion end

"""
    independent_indices(c::AbstractCompletion) -> Vector{Int}

Indices (in `L2OCore.decision_variables` order) of the variables predicted by
the proxy.
"""
function independent_indices end

"""
    dependent_indices(c::AbstractCompletion) -> Vector{Int}

Indices of the variables recovered by the completion.
"""
function dependent_indices end

"""
    equality_completion(c::AbstractCompletion, Z, P; kwargs...) -> Y

The completion `φ_x` of DC3: build `Y` (`n × B`) with `Y[independent, :] = Z`
and the dependent rows chosen so that the equality constraints hold.

Guarantee (`EqualityFeasible`): `h_x(Y) = 0` up to round-off (linear) or the
Newton tolerance (nonlinear, when Newton converges).
"""
function equality_completion end

"""
    completion_pullback(c::AbstractCompletion, Y, P, Ȳ) -> Z̄

Vector-Jacobian product of the completion at the completed point `Y`:
`Z̄ = Ȳ_z + (∂φ/∂z)ᵀ Ȳ_φ`, with `∂φ/∂z = -(J^h_φ)⁻¹ J^h_z` (eq. (3) of
Donti et al.). Used by the correction step to map gradients from `y` to `z`;
differentiable itself, so the correction can be unrolled.
"""
function completion_pullback end

independent_indices(c::AbstractCompletion) = c.independent
dependent_indices(c::AbstractCompletion) = c.dependent

# -----------------------------------------------------------------------------
# Choice of independent / dependent variables
# -----------------------------------------------------------------------------

"""
    select_dependent(J::AbstractMatrix; rtol = 1e-9) -> (rows, dependent)

Choose a maximal set of linearly independent equality rows and as many
dependent variables (columns of `J`) such that `J[rows, dependent]` is
nonsingular, by QR with column pivoting (first on `Jᵀ` for the rows, then on
`J[rows, :]` for the columns). Pivoting picks the best-conditioned columns
greedily; the paper picks a random subset that gives a nonsingular block.
"""
function select_dependent(J::AbstractMatrix; rtol::Real=1e-9)
    m, n = size(J)
    m == 0 && return (Int[], Int[])
    Fr = LinearAlgebra.qr(Matrix(J'), LinearAlgebra.ColumnNorm())
    d = abs.(LinearAlgebra.diag(Fr.R))
    r = isempty(d) ? 0 : count(>(rtol * max(d[1], eps())), d)
    rows = sort(Fr.p[1:r])
    Fc = LinearAlgebra.qr(Matrix(J[rows, :]), LinearAlgebra.ColumnNorm())
    dependent = sort(Fc.p[1:r])
    return rows, dependent
end

function _validate_dependent(dependent, n, r)
    length(dependent) == r || throw(
        ArgumentError(
            "expected $r dependent variables (rank of the equalities), got $(length(dependent))",
        ),
    )
    allunique(dependent) && all(i -> 1 <= i <= n, dependent) ||
        throw(ArgumentError("dependent indices must be unique and within 1:$n"))
    return sort(collect(dependent))
end

function _check_conditioning(JD::AbstractMatrix, what)
    isempty(JD) && return nothing
    κ = LinearAlgebra.cond(Matrix(JD))
    isfinite(κ) && κ < 1e12 || throw(
        ArgumentError(
            "the $what block of the equality Jacobian is singular or ill-conditioned " *
            "(cond = $κ); choose other `dependent` variables",
        ),
    )
    return nothing
end

# -----------------------------------------------------------------------------
# Linear completion
# -----------------------------------------------------------------------------

"""
    LinearCompletion(form::AffineForm; dependent = nothing, rtol = 1e-9)

Closed-form completion for affine equalities `A_E y + B_E p + b_E = ℓ_E`.
With `D` the dependent and `Z` the independent columns,

```math
y_D = -A_{E,D}^{-1}\\,(A_{E,Z}\\, z + B_E\\, p + b_E - \\ell_E),
```

which is precomputed as one affine map `Y = K_z Z + K_p P + k_0`
(`n × n_z`, `n × n_p`, `n`), so completion is a single batched matrix product
(GPU-friendly) and its Jacobian `∂y/∂z = K_z` is exact.

Redundant equality rows are removed (and checked to be consistent for every
parameter value); `dependent` overrides the automatic choice
([`select_dependent`](@ref)).
"""
struct LinearCompletion{M<:AbstractMatrix,V<:AbstractVector} <: AbstractCompletion
    independent::Vector{Int}
    dependent::Vector{Int}
    Kz::M
    Kp::M
    k0::V
end

function LinearCompletion(form::AffineForm{T}; dependent=nothing, rtol::Real=1e-9) where {T}
    E = collect(form.eqidx)
    return _linear_completion(
        Matrix(form.A[E, :]),
        Matrix(form.B[E, :]),
        form.b[E] .- form.lower[E],
        T;
        dependent=dependent,
        rtol=rtol,
    )
end

"""
    LinearCompletion(form::NLPForm; dependent = nothing, rtol = 1e-9)

Linear completion for a problem whose *equality* constraints are affine while
the objective or the inequalities are nonlinear (e.g. the non-convex QP of
Section 4.2 of Donti et al.). The affine data are read from the constant
Jacobian of the equality rows.
"""
function LinearCompletion(form::NLPForm; dependent=nothing, rtol::Real=1e-9)
    E = form.eqidx
    all(view(form.affine, E)) || throw(
        ArgumentError(
            "LinearCompletion needs affine equality constraints; use NewtonCompletion"
        ),
    )
    y, p = zeros(form.n), zeros(form.np)
    J = constraint_jacobian(form, y, p)[E, :]
    F0 = _constraint_values(form, reshape(y, :, 1), reshape(p, :, 1))[E, 1]
    return _linear_completion(
        J[:, 1:(form.n)],
        J[:, (form.n + 1):end],
        F0 .- form.lower[E],
        Float64;
        dependent=dependent,
        rtol=rtol,
    )
end

# Completion of `AE y + BE p + bE = 0`.
function _linear_completion(
    AE::AbstractMatrix, BE::AbstractMatrix, bE::AbstractVector, ::Type{T}; dependent, rtol
) where {T}
    n, np = size(AE, 2), size(BE, 2)
    rows, auto = select_dependent(AE; rtol=rtol)
    _check_redundant_rows(AE, BE, bE, rows, rtol)
    D = dependent === nothing ? auto : _validate_dependent(dependent, n, length(rows))
    Zi = setdiff(1:n, D)
    AD = AE[rows, D]
    _check_conditioning(AD, "dependent")
    Kz = zeros(T, n, length(Zi))
    Kp = zeros(T, n, np)
    k0 = zeros(T, n)
    for (j, i) in enumerate(Zi)
        Kz[i, j] = one(T)
    end
    if !isempty(D)
        Fd = LinearAlgebra.lu(AD)
        # eq. (3) of Donti et al. for affine h: ∂φ/∂z = -(J^h_φ)⁻¹ J^h_z.
        Kz[D, :] .= -(Fd \ AE[rows, Zi])
        Kp[D, :] .= -(Fd \ BE[rows, :])
        k0[D] .= -(Fd \ bE[rows])
    end
    return LinearCompletion(collect(Zi), D, Kz, Kp, k0)
end

# Dropped rows of [A_E B_E b_E] must be combinations of the kept rows, otherwise
# the equalities are infeasible for some parameter values.
function _check_redundant_rows(AE, BE, bE, rows, rtol)
    dropped = setdiff(axes(AE, 1), rows)
    isempty(dropped) && return nothing
    M = hcat(AE, BE, bE)
    K = M[rows, :]
    R = M[dropped, :]
    C = R / K
    resid = maximum(abs, R .- C * K; init=zero(eltype(M)))
    scale = max(one(eltype(M)), maximum(abs, M; init=zero(eltype(M))))
    resid <= sqrt(rtol) * scale || throw(
        ArgumentError(
            "the affine equality constraints are linearly dependent but not consistent " *
            "for all parameter values (residual $resid); the problem is infeasible for " *
            "some parameters",
        ),
    )
    return nothing
end

function equality_completion(
    c::LinearCompletion, Z::AbstractMatrix, P::AbstractMatrix; kwargs...
)
    return c.Kz * Z .+ c.Kp * P .+ c.k0
end

function completion_pullback(
    c::LinearCompletion, Y::AbstractMatrix, P::AbstractMatrix, Ȳ::AbstractMatrix
)
    return c.Kz' * Ȳ
end

# -----------------------------------------------------------------------------
# Newton completion
# -----------------------------------------------------------------------------

"""
    NewtonCompletion(form::NLPForm, problem; dependent = nothing, y0 = start values,
                     maxiter = 50, tol = 1e-10, rtol = 1e-9)

Completion for nonlinear equalities: for each instance, damped Newton's method
on `h_x([z; y_D]) = 0` in the dependent variables `y_D`, started from `y0[D]`
(or from a warm start passed as `init`). Derivatives follow the implicit
function theorem, eqs. (3)–(4) of Donti et al.:
`∂y_D/∂z = -(J^h_D)⁻¹ J^h_Z` and `∂y_D/∂p = -(J^h_D)⁻¹ J^h_p`.

The dependent set is chosen by [`select_dependent`](@ref) on the equality
Jacobian at `(y0, p0)` (start values and current parameter values) unless
given.

Assumptions: `h` is C² near the solution; `J^h_D` is nonsingular there; the
start point is in Newton's basin of attraction. Instances where Newton does not
reach `tol` are returned as they are (the `EqualityFeasible` check reports
them).
"""
struct NewtonCompletion{F<:NLPForm} <: AbstractCompletion
    form::F
    independent::Vector{Int}
    dependent::Vector{Int}
    y0::Vector{Float64}
    maxiter::Int
    tol::Float64
end

function NewtonCompletion(
    form::NLPForm,
    problem::L2OCore.ParametricProblem;
    dependent=nothing,
    y0::Union{Nothing,AbstractVector}=nothing,
    maxiter::Integer=50,
    tol::Real=1e-10,
    rtol::Real=1e-9,
)
    vars = L2OCore.decision_variables(problem)
    if y0 === nothing
        y0 = [something(JuMP.start_value(v), 0.0) for v in vars]
    end
    p0 = L2OCore.parameter_values(problem)
    E = form.eqidx
    J = constraint_jacobian(form, y0, p0)[E, 1:(form.n)]
    rows, auto = select_dependent(J; rtol=rtol)
    length(rows) == length(E) || throw(
        ArgumentError(
            "the nonlinear equality Jacobian has rank $(length(rows)) < $(length(E)) at the " *
            "start point; remove redundant equalities or pass another `y0`",
        ),
    )
    D = dependent === nothing ? auto : _validate_dependent(dependent, form.n, length(E))
    _check_conditioning(J[:, D], "dependent")
    Zi = setdiff(1:(form.n), D)
    return NewtonCompletion(
        form, collect(Zi), D, Vector{Float64}(y0), Int(maxiter), Float64(tol)
    )
end

function _h!(h, c::NewtonCompletion, y, p)
    f = c.form
    x = _setx!(f, y, p)
    MOI.eval_constraint(f.evaluator, f.gbuf, x)
    @inbounds for (i, r) in enumerate(f.eqidx)
        h[i] = f.gbuf[r] - f.lower[r]
    end
    return h
end

function _newton!(y, c::NewtonCompletion, p)
    D, E = c.dependent, c.form.eqidx
    h = zeros(length(E))
    _h!(h, c, y, p)
    nh = LinearAlgebra.norm(h, Inf)
    for _ in 1:(c.maxiter)
        nh <= c.tol && break
        JD = constraint_jacobian(c.form, y, p)[E, D]
        δ = JD \ h
        yD = y[D]
        t = 1.0
        while true
            y[D] .= yD .- t .* δ
            _h!(h, c, y, p)
            nnew = LinearAlgebra.norm(h, Inf)
            if nnew <= (1 - 1e-4 * t) * nh || t < 1e-6
                nh = nnew
                break
            end
            t /= 2
        end
    end
    return y
end

function _complete(c::NewtonCompletion, Z::AbstractMatrix, P::AbstractMatrix, init)
    n = c.form.n
    size(Z, 1) == length(c.independent) || throw(
        DimensionMismatch("Z has $(size(Z, 1)) rows, expected $(length(c.independent))")
    )
    Y = zeros(n, size(Z, 2))
    y = zeros(n)
    for k in axes(Z, 2)
        y[c.independent] .= view(Z, :, k)
        y[c.dependent] .=
            init === nothing ? view(c.y0, c.dependent) : view(init, c.dependent, k)
        _newton!(y, c, view(P, :, k))
        Y[:, k] .= y
    end
    return Y
end

"""
    equality_completion(c::NewtonCompletion, Z, P; init = nothing)

Newton completion of every column; `init` (`n × B`, optional) warm-starts the
dependent variables (the result does not depend on it when Newton converges to
the same root, and no derivative flows through it).
"""
equality_completion(
    c::NewtonCompletion, Z::AbstractMatrix, P::AbstractMatrix; init=nothing
) = _complete(c, Z, P, init)

function ChainRulesCore.rrule(
    ::typeof(equality_completion),
    c::NewtonCompletion,
    Z::AbstractMatrix,
    P::AbstractMatrix;
    init=nothing,
)
    Y = _complete(c, Z, P, init)
    f = c.form
    E, D, Zi = f.eqidx, c.dependent, c.independent
    function equality_completion_pullback(ΔY)
        Ȳ = ChainRulesCore.unthunk(ΔY)
        Z̄, P̄ = zeros(size(Z)), zeros(size(P))
        for k in axes(Z, 2)
            J = constraint_jacobian(f, view(Y, :, k), view(P, :, k))[E, :]
            # eq. (4) of Donti et al.: w = (J^h_D)⁻ᵀ ȳ_D,  z̄ = ȳ_Z - (J^h_Z)ᵀ w,
            # and likewise p̄ = -(J^h_p)ᵀ w.
            w = J[:, D]' \ Ȳ[D, k]
            Z̄[:, k] .= Ȳ[Zi, k] .- J[:, Zi]' * w
            P̄[:, k] .= .-(J[:, (f.n + 1):end]' * w)
        end
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), Z̄, P̄
    end
    return Y, equality_completion_pullback
end

# W[:, k] = J^h_D(y_k, p_k)⁻ᵀ R[:, k]  (one row per equality).
function _transpose_solve(
    c::NewtonCompletion, Y::AbstractMatrix, P::AbstractMatrix, R::AbstractMatrix
)
    f = c.form
    W = zeros(length(f.eqidx), size(Y, 2))
    for k in axes(Y, 2)
        JD = constraint_jacobian(f, view(Y, :, k), view(P, :, k))[f.eqidx, c.dependent]
        W[:, k] .= JD' \ view(R, :, k)
    end
    return W
end

function ChainRulesCore.rrule(
    ::typeof(_transpose_solve),
    c::NewtonCompletion,
    Y::AbstractMatrix,
    P::AbstractMatrix,
    R::AbstractMatrix,
)
    W = _transpose_solve(c, Y, P, R)
    f = c.form
    E, D = f.eqidx, c.dependent
    function transpose_solve_pullback(ΔW)
        W̄ = ChainRulesCore.unthunk(ΔW)
        Ȳ, P̄, R̄ = zeros(size(Y)), zeros(size(P)), zeros(size(R))
        μ = zeros(f.m)
        v = zeros(f.n + f.np)
        for k in axes(Y, 2)
            y, p = view(Y, :, k), view(P, :, k)
            JD = constraint_jacobian(f, y, p)[E, D]
            # J_Dᵀ w = r  ⇒  r̄ = s := J_D⁻¹ w̄,  x̄ = -(Σᵢ wᵢ ∇²hᵢ(x)) ŝ,
            # where ŝ is s placed on the dependent coordinates of x = [y; p].
            s = JD \ view(W̄, :, k)
            R̄[:, k] .= s
            fill!(μ, 0)
            μ[E] .= view(W, :, k)
            fill!(v, 0)
            v[D] .= s
            _hessian_product!(f.hbuf, f, y, p, v, μ)
            Ȳ[:, k] .= .-view(f.hbuf, 1:(f.n))
            P̄[:, k] .= .-view(f.hbuf, (f.n + 1):(f.n + f.np))
        end
        return ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), Ȳ, P̄, R̄
    end
    return W, transpose_solve_pullback
end

function completion_pullback(
    c::NewtonCompletion, Y::AbstractMatrix, P::AbstractMatrix, Ȳ::AbstractMatrix
)
    f = c.form
    # Z̄ = Ȳ_Z - (J^h_Z)ᵀ (J^h_D)⁻ᵀ Ȳ_D, written with differentiable primitives:
    # constraint_vjp(form, Y, P, Wfull) = J_yᵀ Wfull with W on the equality rows.
    W = _transpose_solve(c, Y, P, Ȳ[c.dependent, :])
    Wfull = _embed_rows(W, f.eqidx, f.m)
    G = Ȳ .- constraint_vjp(f, Y, P, Wfull)
    return G[c.independent, :]
end

# Place the rows of W at positions `rows` of an m-row zero matrix (AD-friendly).
function _embed_rows(W::AbstractMatrix, rows::AbstractVector{Int}, m::Int)
    S = _selection(eltype(W), rows, m)
    return S * W
end

function _selection(::Type{T}, rows::AbstractVector{Int}, m::Int) where {T}
    S = zeros(T, m, length(rows))
    for (j, r) in enumerate(rows)
        S[r, j] = one(T)
    end
    return S
end
ChainRulesCore.@non_differentiable _selection(::Any, ::Any, ::Any)
