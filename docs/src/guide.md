# Package guide

## Getting started

A problem is a JuMP model whose instance data are parameters. Wrap it in an
`L2OCore.ParametricProblem` and bind it to a method with `L2OPipeline`:

```jldoctest guide
julia> using L2ODC3, L2OCore, JuMP

julia> model = Model();

julia> @variable(model, d[1:2] in MOI.Parameter.([1.0, 2.0]));

julia> @variable(model, 0 <= g[1:3] <= 4);

julia> @constraint(model, g[1] + g[2] == d[1]);

julia> @constraint(model, g[2] + g[3] == d[2]);

julia> @objective(model, Min, g[1] + 3g[2] + 2g[3]);

julia> pipe = L2OPipeline(ParametricProblem(model), DC3());

julia> output_dimension(pipe)   # independent variables predicted by the proxy
1

julia> dependent_indices(pipe.decomposition)
2-element Vector{Int64}:
 1
 2
```

A proxy is any callable that maps `num_parameters × B` inputs to
`output_dimension × B` outputs. Training minimizes `loss(pipe, P, proxy(P))`,
and prediction is `complete(pipe, P, proxy(P))`:

```jldoctest guide
julia> P = [1.0 0.5; 2.0 1.5];          # two instances (batch = last dimension)

julia> Z = [0.2 0.1];                    # raw proxy output for both

julia> pred = complete(pipe, P, Z);

julia> maximum(evaluate(pipe, P, pred).max_equality_violation) < 1e-12
true

julia> loss(pipe, P, Z) isa Real
true
```

The training loop is up to you, since the package does not depend on an ML
framework. The [example](examples/dc3_qp.md) trains a Lux network with
Optimisers.jl and Zygote:

```julia
ℓ, back = Zygote.pullback(θ -> loss(pipe, P, proxy(P, θ)), θ)
opt, θ = Optimisers.update(opt, θ, back(one(ℓ))[1])
```

Both methods are self-supervised. `generate_data(pipe; sampler, n)` returns
parameter samples only. Pass `optimizer = HiGHS.Optimizer` (or Ipopt) to also
solve them, so you can compare against the optimum with `evaluate`.

## What you need to provide

- A JuMP model with parameters (`@variable(m, p in MOI.Parameter(v))`) and
  box-like constraint sets (`==`, `<=`, `>=`, intervals, variable bounds,
  `Zeros`, `Nonnegatives`, `Nonpositives`), second-order cones
  (`[t; x] in SecondOrderCone()`) and rotated second-order cones. Other cones
  (PSD, exponential) are rejected (see [Extension points](@ref)).
- Constraint functions and objective that are affine, quadratic or nonlinear
  (`@constraint`/`@objective` with nonlinear expressions, not the legacy `@NL`
  macros).
- For nonlinear problems, a correction `step_size`. With nonlinear equalities,
  start values (`start = ...`) near the feasible manifold: they are the Newton
  initial guess and the point where the dependent variables are chosen.
- A proxy (any callable) and a training loop.

## Concepts

A pipeline `L2OPipeline(ParametricProblem(model), DC3())` holds what is computed once per problem; the proxy maps parameters `P` to independent variables `Z`, the completion gives `Y` with ``h(Y) = 0``, and the correction and the soft loss act on the completed `Y` (see the figure on the [Theory](theory.md) page).

| concept | types | role |
|:--|:--|:--|
| problem form | [`AbstractProblemForm`](@ref): [`AffineForm`](@ref), [`NLPForm`](@ref); [`problem_form`](@ref) picks one | differentiable, batched access to the objective, constraint rows and Jacobian-transpose products; `AffineForm` uses dense matrices and is GPU-ready, `NLPForm` wraps `MathOptInterface.Nonlinear` and evaluates column by column on the CPU |
| completion | [`AbstractCompletion`](@ref): [`LinearCompletion`](@ref), [`NewtonCompletion`](@ref) | the map ``z \mapsto y`` that satisfies the equalities, and its pullback |
| decomposition | [`DC3Decomposition`](@ref), [`FSNetDecomposition`](@ref) | what `L2OCore.decompose` computes once per problem (form, completion, step size), stored in `pipe.decomposition` |
| batch entry points | [`predict`](@ref), [`training_loss`](@ref) | the same computations as `complete` and `loss` on a decomposition directly; use them with device arrays |

## Choosing hyper-parameters

- `step_size`: `nothing` means ``(1 - \mu)/L`` for affine problems, which is
  safe. Without momentum the violation never increases, and with momentum
  ``\mu`` the heavy-ball iterates still converge (see Theory). For nonlinear problems, start
  around ``0.01``–``0.1`` and verify on a few samples that
  [`inequality_violation`](@ref) decreases from one correction step to the next.
- `train_steps` (``t_\text{train}``): a few steps (5–10) are enough to shape the
  gradient. Every step adds a completion and a backward pass.
- `test_steps` with `test_tol`: allow many test-time steps and stop early once
  the inequalities hold to `test_tol`.
- `λ_ineq`, `λ_eq`: 10 each (the paper tunes their sum in ``\{1, 10, 100\}``).
  `penalty = :norm` with `λ_ineq = λ_eq = 5` reproduces the loss of the authors'
  code. Larger values push the proxy toward
  feasibility but make the loss stiffer.
- FSNet: `solver = :lbfgs` reaches feasibility in far fewer iterations than
  gradient descent. On the paper's QP family, 50 L-BFGS iterations bring
  ``\phi`` below ``10^{-12}`` where gradient descent is still above ``10^{-1}``.
  `train_steps` (``K``) and `unroll` (``K'``) trade gradient accuracy for
  cost; `ρ` weighs the distance between the prediction and its feasible
  correction.

## AD and ML frameworks

The package is framework-agnostic. Its functions are plain array code or have
ChainRules rules, and they never call an AD system themselves (no nested AD).

| path | Zygote | ForwardDiff | Enzyme / Mooncake |
|:--|:--|:--|:--|
| affine problems (`AffineForm`, `LinearCompletion`) | tested | tested | not tested (plain array code, expected to work) |
| nonlinear problems (`NLPForm`, `NewtonCompletion`) | tested (ChainRules rules) | not supported (Float64 buffers) | not supported (rules not imported) |

Proxies: the example uses Lux (tested in the documentation build). Flux models
are callables and work the same way. Derivative (Sobolev) supervision needs
nested AD for the input gradients of the proxy. That belongs to the proxy and
training loop and is handled by `L2OCore.AbstractMethodWrapper` add-ons, not by
this package.

## GPU

The affine path uses only matrix products and broadcasts. Move the
decomposition with Adapt and call the batch entry points on device arrays:

```julia
using Adapt, CUDA
d = adapt(CuArray{Float32}, pipe.decomposition)   # Float32 on the device
P, Z = cu(P), proxy_gpu(cu(P))
Y = predict(pipe.method, d, P, Z)                 # completion + correction on the GPU
ℓ = training_loss(pipe.method, d, P, Z)
```

The test suite runs the same code on `JLArrays.JLArray`, a CPU-backed GPU array
type that forbids scalar indexing, so device compatibility is checked on CI
without a GPU. [`NLPForm`](@ref) is CPU-only. Batched GPU kernels for nonlinear
problems are the role of a future form built from ExaModels (see below).

## Extension points

All extension is by dispatch on abstract types:

- **New completion** (e.g. a structured solve, or a closed-form completion for
  power-flow equations): subtype [`AbstractCompletion`](@ref) and implement
  [`equality_completion`](@ref), [`completion_pullback`](@ref),
  `independent_indices` and `dependent_indices`. Make both differentiable, with
  rules where needed. Pass it as `DC3(; completion = (problem, form) -> MyCompletion(...))`.
- **New problem form** (e.g. batched GPU kernels): subtype
  [`AbstractProblemForm`](@ref) and implement
  `L2OCore.batch_objective`, `L2OCore.constraint_values`,
  `L2OCore.equality_rows`, `L2OCore.num_decision_variables`,
  [`row_bounds`](@ref), [`equality_indices`](@ref),
  [`inequality_indices`](@ref), [`objective_sign`](@ref) and
  [`constraint_vjp`](@ref). DC3 and FSNet then work unchanged. A form is also an
  `L2OCore.AbstractEvaluator`, so `evaluate` accepts it as `evaluator`.

### Planned: ExaModels and other front ends

Problems written in ExaModels already carry batched, GPU-compiled kernels for
the objective, constraints, Jacobian-vector and Hessian-vector products. A
form that wraps them (an `AbstractProblemForm` in a package extension on
ExaModels) gives DC3 and FSNet a batched GPU path for nonlinear problems through
the same interface. A JuMP model can reach the same kernels through ExaModels'
MathOptInterface support. Only `constraint_vjp` and its derivative (a
Hessian-vector product) are needed beyond plain evaluation.

## The LearningToOptimize ecosystem

- [LearningToOptimize.jl](https://github.com/LearningToOptimize/LearningToOptimize.jl):
  the core interface (`L2OCore`: problems, pipelines, evaluators),
  basic penalty methods and dataset utilities.
- [L2ODLL.jl](https://github.com/LearningToOptimize/L2ODLL.jl): dual Lagrangian
  learning (dual proxies with valid bounds).
- [DecisionRules.jl](https://github.com/LearningToOptimize/DecisionRules.jl):
  policies for multistage stochastic programs.
- L2ODC3.jl (this package): primal proxies with hard equality constraints (DC3)
  and feasibility seeking (FSNet).
