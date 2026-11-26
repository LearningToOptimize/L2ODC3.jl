# The DC3 and FSNet methods and their L2OCore pipeline steps.

const _SUPPORTED_CLASSES = Union{
    L2OCore.LinearConic,L2OCore.ConvexQuadratic,L2OCore.SmoothNonlinear
}

_asbatch(x::AbstractMatrix) = x
_asbatch(x::AbstractVector) = reshape(x, :, 1)
_mean(v::AbstractVector) = sum(v) / length(v)

# Hyper-parameters in the precision of the decomposition (e.g. Float32 after
# `Adapt.adapt(CuArray{Float32}, d)`), so no Float64 promotion on the device.
_like(d, x::Real) = convert(typeof(d.step_size), x)

# =============================================================================
# DC3
# =============================================================================

"""
    DC3(; λ_ineq = 10.0, λ_eq = 10.0, penalty = :squared, train_steps = 10,
        test_steps = 10, step_size = nothing, momentum = 0.5, test_tol = 0.0,
        completion = :auto, dependent = nothing,
        newton_maxiter = 50, newton_tol = 1e-10)

Deep Constraint Completion and Correction (Donti, Rolnick & Kolter, ICLR 2021,
[arXiv:2104.12225](https://arxiv.org/abs/2104.12225)), a primal method.

The proxy predicts the independent variables `z` (`output_dimension(pipe)`
rows). `complete(pipe, inputs, output)` applies the equality completion
`φ_x` ([`equality_completion`](@ref)) and `test_steps` correction steps
([`inequality_correction`](@ref)); `loss(pipe, inputs, output)` applies
`train_steps` correction steps and returns the soft loss (eq. (2) of the paper)

```math
\\ell = \\frac1B\\sum_{k} f_x(\\hat y_k) + \\lambda_g \\|\\mathrm{ReLU}(g_x(\\hat y_k))\\|_2^2
       + \\lambda_h \\|h_x(\\hat y_k)\\|_2^2 .
```

Keyword arguments:

  - `λ_ineq`, `λ_eq`: soft-loss weights `λ_g`, `λ_h`;
  - `penalty`: `:squared` (eq. (2) of the paper, default) or `:norm`, which
    uses `‖ReLU(g)‖₂` and `‖h‖₂` without squares as the reference implementation
    does (it also splits a total weight 10 as `λ_g = λ_h = 5`);
  - `train_steps`, `test_steps`: correction steps `t_train`, `t_test`;
  - `step_size`: correction step `γ`; `nothing` uses `(1 - momentum)/L` for
    affine problems ([`default_step_size`](@ref)), a step for which both plain
    gradient descent (`momentum = 0`) and heavy-ball momentum provably converge
    on the convex reduced violation; it must be given for nonlinear problems.
    (The paper fixes `γ = 10⁻⁷` for its QPs, Appendix B.1.);
  - `momentum`: heavy-ball momentum `μ` of the correction (0.5 in the paper,
    Appendix B; the descent inequality of the Theory page is stated for
    `μ = 0`);
  - `test_tol`: stop the test-time correction once `max ReLU(g) ≤ test_tol`
    (the paper's stopping tolerance is `10⁻⁴`, Appendix B; `0` runs exactly
    `t_test` steps as in Algorithm 1);
  - `completion`: `:auto` (closed-form [`LinearCompletion`](@ref) when the
    equalities are affine, [`NewtonCompletion`](@ref) otherwise), `:linear`,
    `:newton` (forces the nonlinear form and Newton), or a function
    `(problem, form) -> AbstractCompletion` (extension point);
  - `dependent`: indices of the dependent variables (default: chosen by
    [`select_dependent`](@ref));
  - `newton_maxiter`, `newton_tol`: Newton completion settings.

Completed predictions satisfy the equality constraints to round-off for
affine equalities, and to `newton_tol` when Newton converges for nonlinear
equalities. Supports `LinearConic` (box-like sets and second-order cones),
`ConvexQuadratic` and
`SmoothNonlinear` problems.
"""
struct DC3{S,C} <: L2OCore.AbstractPrimalMethod
    λ_ineq::Float64
    λ_eq::Float64
    penalty::Symbol
    train_steps::Int
    test_steps::Int
    step_size::S
    momentum::Float64
    test_tol::Float64
    completion::C
    dependent::Union{Nothing,Vector{Int}}
    newton_maxiter::Int
    newton_tol::Float64
end

function DC3(;
    λ_ineq::Real=10.0,
    λ_eq::Real=10.0,
    penalty::Symbol=:squared,
    train_steps::Integer=10,
    test_steps::Integer=10,
    step_size::Union{Nothing,Real}=nothing,
    momentum::Real=0.5,
    test_tol::Real=0.0,
    completion=:auto,
    dependent::Union{Nothing,AbstractVector{<:Integer}}=nothing,
    newton_maxiter::Integer=50,
    newton_tol::Real=1e-10,
)
    train_steps >= 0 && test_steps >= 0 || throw(ArgumentError("steps must be nonnegative"))
    step_size === nothing ||
        step_size > 0 ||
        throw(ArgumentError("step_size must be positive"))
    0 <= momentum < 1 || throw(ArgumentError("momentum must be in [0, 1)"))
    penalty in (:squared, :norm) ||
        throw(ArgumentError("penalty must be :squared or :norm"))
    completion isa Symbol &&
        !(completion in (:auto, :linear, :newton)) &&
        throw(ArgumentError("completion must be :auto, :linear, :newton or a function"))
    return DC3(
        Float64(λ_ineq),
        Float64(λ_eq),
        penalty,
        Int(train_steps),
        Int(test_steps),
        step_size === nothing ? nothing : Float64(step_size),
        Float64(momentum),
        Float64(test_tol),
        completion,
        dependent === nothing ? nothing : collect(Int, dependent),
        Int(newton_maxiter),
        Float64(newton_tol),
    )
end

"""
    DC3Decomposition(form, completion, step_size)

Result of `L2OCore.decompose(problem, ::DC3)`: the differentiable problem
[`AbstractProblemForm`](@ref), the [`AbstractCompletion`](@ref) (which fixes the
independent variables predicted by the proxy) and the resolved correction step
size. `output_dimension(d)` is the number of independent variables.

Move it to a GPU with `Adapt.adapt(CuArray, d)` (affine problems) and call
[`predict`](@ref) / [`training_loss`](@ref) with device arrays.
"""
struct DC3Decomposition{F<:AbstractProblemForm,C<:AbstractCompletion,T<:Real} <:
       L2OCore.AbstractDecomposition
    form::F
    completion::C
    step_size::T
end

L2OCore.output_dimension(d::DC3Decomposition) = length(independent_indices(d.completion))
independent_indices(d::DC3Decomposition) = independent_indices(d.completion)
dependent_indices(d::DC3Decomposition) = dependent_indices(d.completion)

L2OCore.supports(::DC3, ::_SUPPORTED_CLASSES) = true

"""
    default_step_size(form, completion) -> Real
    default_step_size(form) -> Real

`1/L`, with `L` the Lipschitz constant of the gradient of the violation that is
descended: the reduced inequality violation `v(z)` for DC3
(`L = 2‖A_I K_z‖₂²`) or FSNet's `φ(y)` (`L = 2‖A‖₂²`). Only defined for
[`AffineForm`](@ref); nonlinear problems need an explicit `step_size`.
`DC3` scales it by `1 - momentum` (see [`DC3`](@ref)).
"""
function default_step_size(form::AffineForm, c::LinearCompletion)
    I = collect(form.gidx)
    isempty(I) && return one(eltype(form.A))
    L = 2 * LinearAlgebra.opnorm(Matrix(form.A[I, :] * c.Kz))^2
    return L > 0 ? inv(L) : one(L)
end

function default_step_size(form::AffineForm)
    L = lipschitz_constant(form)
    return L > 0 ? inv(L) : one(L)
end

function default_step_size(form::AbstractProblemForm, args...)
    throw(
        ArgumentError(
            "no default step size for $(nameof(typeof(form))): the violation has no global " *
            "Lipschitz constant; pass `step_size`",
        ),
    )
end

function _dc3_completion(m::DC3, problem, form)
    m.completion isa Symbol || return m.completion(problem, form)
    form isa AffineForm && return LinearCompletion(form; dependent=m.dependent)
    affine_eq = all(view(form.affine, form.eqidx))
    if m.completion === :linear || (m.completion === :auto && affine_eq)
        affine_eq ||
            throw(ArgumentError("completion = :linear needs affine equality constraints"))
        return LinearCompletion(form; dependent=m.dependent)
    end
    return NewtonCompletion(
        form, problem; dependent=m.dependent, maxiter=m.newton_maxiter, tol=m.newton_tol
    )
end

function L2OCore.decompose(problem::L2OCore.ParametricProblem, m::DC3)
    form = m.completion === :newton ? NLPForm(problem) : problem_form(problem)
    c = _dc3_completion(m, problem, form)
    # Default (1 - μ)/L: the descent lemma for μ = 0; for μ > 0 a step in the
    # range where averaged heavy-ball iterates converge on convex L-smooth
    # functions (Ghadimi, Feyzmahdavian & Johansson, 2015).
    γ = if m.step_size === nothing
        (1 - m.momentum) * default_step_size(form, c)
    else
        m.step_size
    end
    return DC3Decomposition(form, c, γ)
end

"""
    predict(method, decomposition, P, output; training = false) -> Y

Map a batch of raw proxy outputs to primal points without an `L2OPipeline`
(useful for GPU arrays): for [`DC3`](@ref), completion plus `train_steps` or
`test_steps` corrections; for [`FSNet`](@ref), the truncated feasibility-seeking
map.
"""
function predict(
    m::DC3, d::DC3Decomposition, P::AbstractMatrix, Z::AbstractMatrix; training::Bool=false
)
    _, Y = inequality_correction(
        d.form,
        d.completion,
        Z,
        P;
        steps=training ? m.train_steps : m.test_steps,
        step_size=d.step_size,
        momentum=_like(d, m.momentum),
        tol=training ? 0.0 : m.test_tol,
    )
    return Y
end

"""
    soft_loss(form, Y, P; λ_ineq, λ_eq, penalty = :squared) -> Real

Batch mean of the soft loss of eq. (2) of Donti et al.,
`f_x(y) + λ_ineq ‖ReLU(g_x(y))‖²₂ + λ_eq ‖h_x(y)‖²₂` (objective in minimization
form). With `penalty = :norm` the penalties are `‖ReLU(g_x(y))‖₂` and
`‖h_x(y)‖₂` (not squared), as in the reference implementation of the paper;
the norm is smoothed as `sqrt(‖r‖² + ε²)` with `ε = eps(T)` so that its
gradient is finite at feasible points.
"""
function soft_loss(
    form::AbstractProblemForm,
    Y::AbstractMatrix,
    P::AbstractMatrix;
    λ_ineq::Real,
    λ_eq::Real,
    penalty::Symbol=:squared,
)
    penalty in (:squared, :norm) ||
        throw(ArgumentError("penalty must be :squared or :norm"))
    pen = penalty === :squared ? identity : _smooth_sqrt
    ℓ = min_objective(form, Y, P)
    if λ_ineq != 0
        ℓ = ℓ .+ λ_ineq .* pen(inequality_violation(form, Y, P))
    end
    if λ_eq != 0
        ℓ = ℓ .+ λ_eq .* pen(equality_violation(form, Y, P))
    end
    return _mean(ℓ)
end

_smooth_sqrt(v::AbstractVector) = sqrt.(v .+ eps(eltype(v))^2)

"""
    training_loss(method, decomposition, P, output) -> Real

The training loss of a batch of raw proxy outputs without an `L2OPipeline`
(see [`predict`](@ref)).
"""
function training_loss(m::DC3, d::DC3Decomposition, P::AbstractMatrix, Z::AbstractMatrix)
    Y = predict(m, d, P, Z; training=true)
    return soft_loss(
        d.form, Y, P; λ_ineq=_like(d, m.λ_ineq), λ_eq=_like(d, m.λ_eq), penalty=m.penalty
    )
end

function L2OCore.complete(m::DC3, pipe::L2OCore.L2OPipeline, inputs, output)
    Y = predict(m, pipe.decomposition, _asbatch(inputs), _asbatch(output); training=false)
    return L2OCore.Prediction(; primal=Y)
end

function L2OCore.loss(m::DC3, pipe::L2OCore.L2OPipeline, inputs, output; kwargs...)
    return training_loss(m, pipe.decomposition, _asbatch(inputs), _asbatch(output))
end

# =============================================================================
# FSNet
# =============================================================================

"""
    FSNet(; ρ = 10.0, train_steps = 20, test_steps = 100, unroll = train_steps,
          step_size = nothing, penalty = 10.0, penalty_threshold = 1e3,
          test_tol = 0.0, solver = :gd, memory = 10)

Feasibility-Seeking Neural Network (Nguyen & Donti, 2025,
[arXiv:2506.00362](https://arxiv.org/abs/2506.00362)), a primal method. The proxy
predicts all decision variables `y_θ(x)`; [`feasibility_seeking`](@ref) runs
gradient descent on the violation `φ = ‖h‖² + ‖ReLU(g)‖²` from there, and the
loss is (eq. (6) of the paper, with the early-stage penalty of Section 5)

```math
F = f_x(\\hat y) + \\frac{\\rho}{2}\\|y_\\theta(x) - \\hat y\\|_2^2
    + \\rho_\\phi\\, \\phi(y_\\theta(x))\\,[\\phi(y_\\theta(x)) \\ge Q].
```

Keyword arguments: `ρ`; `train_steps` (`K`) and `unroll` (`K'`, derivatives
flow through the first `K'` of the `K` iterations); `test_steps`; `step_size`
(`η_φ`; `nothing` uses `1/L` for affine problems); `penalty` (`ρ_φ`; `0`
disables it) and `penalty_threshold` (`Q`); `test_tol` (early stop at test time).
The defaults `ρ = 10` (distance weight `ρ/2 = 5`), `ρ_φ = 10` and `Q = 10³`
are those of the authors' reference configuration; the switch is evaluated per
instance on `φ(y_θ(x))`, as in the paper's formula (the reference code switches
on batch means).

`solver = :gd` (default) runs gradient descent with step `η_φ`, the setting of
the paper's convergence theory (Theorem 1). `solver = :lbfgs` runs batched
L-BFGS with `memory` curvature pairs and Armijo backtracking
([`feasibility_seeking_lbfgs`](@ref)), the inner solver of the paper's
experiments; `step_size` is then only the first trial step (default `1/L` for
affine problems, `1` otherwise).

A finite number of steps does not make the prediction feasible; under the
assumptions of [`feasibility_seeking`](@ref) each step reduces the violation.
"""
struct FSNet{S} <: L2OCore.AbstractPrimalMethod
    ρ::Float64
    train_steps::Int
    test_steps::Int
    unroll::Int
    step_size::S
    penalty::Float64
    penalty_threshold::Float64
    test_tol::Float64
    solver::Symbol
    memory::Int
end

function FSNet(;
    ρ::Real=10.0,
    train_steps::Integer=20,
    test_steps::Integer=100,
    unroll::Integer=train_steps,
    step_size::Union{Nothing,Real}=nothing,
    penalty::Real=10.0,
    penalty_threshold::Real=1e3,
    test_tol::Real=0.0,
    solver::Symbol=:gd,
    memory::Integer=10,
)
    solver in (:gd, :lbfgs) || throw(ArgumentError("solver must be :gd or :lbfgs"))
    memory >= 1 || throw(ArgumentError("memory must be at least 1"))
    train_steps >= 0 && test_steps >= 0 && unroll >= 0 ||
        throw(ArgumentError("steps must be nonnegative"))
    step_size === nothing ||
        step_size > 0 ||
        throw(ArgumentError("step_size must be positive"))
    return FSNet(
        Float64(ρ),
        Int(train_steps),
        Int(test_steps),
        Int(unroll),
        step_size === nothing ? nothing : Float64(step_size),
        Float64(penalty),
        Float64(penalty_threshold),
        Float64(test_tol),
        solver,
        Int(memory),
    )
end

"""
    FSNetDecomposition(form, step_size)

Result of `L2OCore.decompose(problem, ::FSNet)`: the proxy predicts all
decision variables (`output_dimension = L2OCore.num_decision_variables(form)`).
"""
struct FSNetDecomposition{F<:AbstractProblemForm,T<:Real} <: L2OCore.AbstractDecomposition
    form::F
    step_size::T
end

L2OCore.output_dimension(d::FSNetDecomposition) = L2OCore.num_decision_variables(d.form)
L2OCore.supports(::FSNet, ::_SUPPORTED_CLASSES) = true

function L2OCore.decompose(problem::L2OCore.ParametricProblem, m::FSNet)
    form = problem_form(problem)
    γ = if m.step_size !== nothing
        m.step_size
    elseif m.solver === :lbfgs && !(form isa AffineForm)
        1.0  # only the first trial step of the line search
    else
        default_step_size(form)
    end
    return FSNetDecomposition(form, γ)
end

function predict(
    m::FSNet,
    d::FSNetDecomposition,
    P::AbstractMatrix,
    Y::AbstractMatrix;
    training::Bool=false,
)
    steps = training ? m.train_steps : m.test_steps
    unroll = training ? m.unroll : 0
    tol = training ? 0.0 : m.test_tol
    if m.solver === :lbfgs
        return feasibility_seeking_lbfgs(
            d.form, Y, P; steps, unroll, memory=m.memory, step_size=d.step_size, tol
        )
    end
    return feasibility_seeking(d.form, Y, P; steps, unroll, step_size=d.step_size, tol)
end

function training_loss(
    m::FSNet, d::FSNetDecomposition, P::AbstractMatrix, Y::AbstractMatrix
)
    Ŷ = predict(m, d, P, Y; training=true)
    ℓ = min_objective(d.form, Ŷ, P) .+ _like(d, m.ρ / 2) .* _colsumsq(Y .- Ŷ)
    if m.penalty != 0
        φ = total_violation(d.form, Y, P)
        on = _stopgrad(φ .>= m.penalty_threshold)
        ℓ = ℓ .+ _like(d, m.penalty) .* on .* φ
    end
    return _mean(ℓ)
end

function L2OCore.complete(m::FSNet, pipe::L2OCore.L2OPipeline, inputs, output)
    Y = predict(m, pipe.decomposition, _asbatch(inputs), _asbatch(output); training=false)
    return L2OCore.Prediction(; primal=Y)
end

function L2OCore.loss(m::FSNet, pipe::L2OCore.L2OPipeline, inputs, output; kwargs...)
    return training_loss(m, pipe.decomposition, _asbatch(inputs), _asbatch(output))
end

# =============================================================================
# Data generation (both methods are self-supervised)
# =============================================================================

"""
    generate_data(method::Union{DC3,FSNet}, pipe; sampler, n, rng = Random.default_rng(),
                  optimizer = nothing, kwargs...) -> L2ODataset

DC3 and FSNet are self-supervised: training needs only parameter samples. Without
`optimizer` the dataset holds inputs only (no solver calls). With `optimizer`,
every instance is also solved (`L2OCore.solve_instances`) so that the primal
labels and optimal objectives are available for evaluation. The metadata record
the method and, for DC3, the independent variables.
"""
function L2OCore.generate_data(
    m::Union{DC3,FSNet},
    pipe::L2OCore.L2OPipeline;
    sampler,
    n::Integer,
    rng::Random.AbstractRNG=Random.default_rng(),
    optimizer=nothing,
    kwargs...,
)
    P = L2OCore.sample_parameters(sampler, rng, n)
    ds = if optimizer === nothing
        L2OCore.L2ODataset(Matrix{Float64}(P))
    else
        L2OCore.solve_instances(pipe.problem, P; optimizer=optimizer, primal=true, kwargs...)
    end
    ds.metadata["method"] = string(nameof(typeof(m)))
    if pipe.decomposition isa DC3Decomposition
        ds.metadata["independent"] = independent_indices(pipe.decomposition)
    end
    return ds
end
