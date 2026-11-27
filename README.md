<div align="center">

# L2ODC3.jl

<p><sub><b>EQUALITY COMPLETION · INEQUALITY CORRECTION · HARD CONSTRAINTS</b></sub></p>

[![CI](https://github.com/LearningToOptimize/L2ODC3.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/LearningToOptimize/L2ODC3.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://LearningToOptimize.github.io/L2ODC3.jl/dev/) [![Codecov](https://codecov.io/gh/LearningToOptimize/L2ODC3.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/LearningToOptimize/L2ODC3.jl) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) [![JET](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

</div>

<p align="center">
  <picture>
    <source media="(max-width: 640px)" srcset="docs/src/assets/diagrams/l2odc3--completion-correction-compact.svg">
    <img src="docs/src/assets/diagrams/l2odc3--completion-correction.svg" width="760" alt="DC3, deep constraint completion and correction. Forward pass: the proxy N_theta maps the parameters x to the independent variables z; the equality completion phi_x of z solves h_x of z and phi_x of z equal to zero (one matrix product for affine equalities, Newton's method otherwise); t unrolled correction steps, z replaced by z minus gamma Delta z and re-completed, reduce the inequality violation (t_train steps in training, t_test at inference), giving y-hat. Training loss (eq. 2 of DC3): ell of y-hat equals f_x of y-hat plus lambda_g times the squared norm of ReLU of g_x of y-hat plus lambda_h times the squared norm of h_x of y-hat; self-supervised, no solver labels. Backward pass: gradients return to theta through the unrolled corrections and, by the implicit function theorem, through the completion, d phi_x / dz equals minus the inverse of J^h with respect to phi times J^h with respect to z (eq. 3).">
  </picture>
</p>
<p align="center"><i>The proxy predicts the independent variables; the equalities are solved for the rest, unrolled correction steps reduce the inequality violation, and the soft loss trains the proxy through both.</i></p>

L2ODC3.jl trains neural-network proxies for parametric optimization problems with **hard constraints**. It implements **DC3**, *Deep Constraint Completion and Correction* ([Donti, Rolnick & Kolter, ICLR 2021](https://arxiv.org/abs/2104.12225)). The proxy predicts only the *independent* variables. The equality constraints are solved for the rest, in closed form for affine equalities and by Newton's method with implicit-function-theorem derivatives otherwise, so every prediction satisfies them exactly. Unrolled gradient steps along the equality manifold then remove the inequality violation. Training is self-supervised: the loss is the objective plus a soft penalty, and no solver labels are needed. The package also implements **FSNet** ([Nguyen & Donti, 2025](https://arxiv.org/abs/2506.00362)), which predicts every variable and descends the total violation `‖h‖² + ‖ReLU(g)‖²` to a feasible point. Problems are written in JuMP. The package works with any proxy and any ChainRules-compatible AD, and affine problems run on GPU arrays.

## Installation

L2ODC3 builds on `L2OCore`, the shared interface of the
[LearningToOptimize](https://github.com/LearningToOptimize) ecosystem, which
lives in the `lib/L2OCore` folder of LearningToOptimize.jl. The registered
LearningToOptimize 1.0.0 predates this API and does not contain `L2OCore`.
Until both packages are registered, install `L2OCore` first and then L2ODC3,
on any Julia version from 1.10:

```julia
using Pkg
Pkg.add(url="https://github.com/LearningToOptimize/LearningToOptimize.jl", subdir="lib/L2OCore")
Pkg.add(url="https://github.com/LearningToOptimize/L2ODC3.jl")
```

The example below also needs JuMP and Zygote: `Pkg.add(["JuMP", "Zygote"])`.

## Example

```julia
using L2ODC3, L2OCore, JuMP, Zygote

# Parametric QP: x are the parameters, y the decisions.
model = Model()
@variable(model, x[1:2] in MOI.Parameter.(zeros(2)))
@variable(model, -1 <= y[1:4] <= 1)
@constraint(model, y[1] + y[2] + y[3] == x[1])
@constraint(model, y[2] - y[4] == x[2])
@objective(model, Min, sum(y .^ 2) + y[1] - y[3])

pipe = L2OPipeline(ParametricProblem(model), DC3())

# Any callable is a proxy; here an affine map z = W x + b trained by gradient descent.
W, b = zeros(output_dimension(pipe), 2), zeros(output_dimension(pipe))
for _ in 1:300
    X = rand(2, 64) .- 0.5
    gW, gb = Zygote.gradient((W, b) -> loss(pipe, X, W * X .+ b), W, b)
    W .-= 0.1 .* gW
    b .-= 0.1 .* gb
end

X = rand(2, 5) .- 0.5
pred = complete(pipe, X, W * X .+ b)   # completion + correction
report = evaluate(pipe, X, pred)       # objective and violations per instance
maximum(report.max_equality_violation) # equalities hold to round-off
```

The [documentation](https://LearningToOptimize.github.io/L2ODC3.jl/dev/) has a
complete example. It trains a Lux network with DC3 and with FSNet on the convex
QP of the DC3 paper and compares both with HiGHS.

## What you need to provide

- A **JuMP model with parameters** (`@variable(m, p in MOI.Parameter(v))`). The
  parameters are the proxy inputs. Constraints use box-like sets (`==`, `<=`,
  `>=`, intervals, variable bounds) or second-order cones; functions may be
  affine, quadratic or nonlinear.
- A **proxy**: any callable from `num_parameters × B` to
  `output_dimension(pipe) × B` (a Lux or Flux network, a closure, an affine map).
- A **training loop** in the framework of your choice, minimizing
  `loss(pipe, X, proxy(X))`.
- For nonlinear problems, a correction `step_size` and start values for the
  Newton completion.

## What you get

| | DC3 | FSNet |
|:--|:--|:--|
| proxy predicts | independent variables | all variables |
| equalities | exact (round-off for affine, Newton tolerance otherwise) | approximate, driven to 0 |
| inequalities | unrolled correction on the equality manifold | unrolled descent on the total violation |
| problem classes | LP, QP, SOCP, smooth NLP | LP, QP, SOCP, smooth NLP |
| GPU (affine problems) | yes (`Adapt.adapt(CuArray, pipe.decomposition)`) | yes |

The DC3 completion satisfies affine equalities to round-off, and its
implicit-function-theorem derivatives are exact. For steps `≤ 1/L`, the
correction and FSNet's feasibility seeking never increase the violation, and
for affine constraints with a nonempty feasible set they drive it to zero. The
[Theory](https://LearningToOptimize.github.io/L2ODC3.jl/dev/theory/) page
derives these properties and compares the defaults with the papers.

## The LearningToOptimize ecosystem

| Package | What it provides |
|:--|:--|
| [LearningToOptimize.jl](https://github.com/LearningToOptimize/LearningToOptimize.jl) | the penalty method, Flux training and solver-labeled dataset generation; hosts the two subpackages below |
| ↳ [L2OCore](https://github.com/LearningToOptimize/LearningToOptimize.jl/tree/main/lib/L2OCore) | the shared interface: method families, problem wrapper, batched evaluators, datasets |
| ↳ [L2OSobolev](https://github.com/LearningToOptimize/LearningToOptimize.jl/tree/main/lib/L2OSobolev) | derivative (Sobolev) supervision added to any method |
| [L2ODLL.jl](https://github.com/LearningToOptimize/L2ODLL.jl) | Dual Lagrangian Learning: dual proxies that give valid dual bounds for conic problems |
| [L2ODC3.jl](https://github.com/LearningToOptimize/L2ODC3.jl) | DC3: equality completion and inequality correction for primal proxies |
| [L2OALM.jl](https://github.com/LearningToOptimize/L2OALM.jl) | primal-dual learning with an augmented Lagrangian |

## Citation

If you use this package, please cite the methods it implements (and this
package, see [`CITATION.cff`](CITATION.cff)):

```bibtex
@inproceedings{donti2021dc3,
  title     = {{DC3}: A learning method for optimization with hard constraints},
  author    = {Donti, Priya L. and Rolnick, David and Kolter, J. Zico},
  booktitle = {International Conference on Learning Representations},
  year      = {2021},
  url       = {https://arxiv.org/abs/2104.12225},
}
@article{nguyen2025fsnet,
  title   = {{FSNet}: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees},
  author  = {Nguyen, Hoang T. and Donti, Priya L.},
  journal = {arXiv preprint arXiv:2506.00362},
  year    = {2025},
}
```
