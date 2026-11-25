# Theory

This page states the two methods as optimization problems, fixes the notation,
and lists every claim the implementation makes, together with its assumptions
and the test that checks it numerically. Equation numbers in parentheses refer
to the papers:

- **[DC3]** P. L. Donti, D. Rolnick, J. Z. Kolter. *DC3: A learning method for
  optimization with hard constraints.* ICLR 2021.
  [arXiv:2104.12225](https://arxiv.org/abs/2104.12225)
- **[FSNet]** H. T. Nguyen, P. L. Donti. *FSNet: Feasibility-Seeking Neural
  Network for Constrained Optimization with Guarantees.* 2025.
  [arXiv:2506.00362](https://arxiv.org/abs/2506.00362)

## The parametric problem

For parameters ``x \in \mathbb{R}^d`` (JuMP parameters, the columns of `P`)
both papers consider (eq. (1) of [DC3] and of [FSNet])

```math
\min_{y \in \mathbb{R}^n} \; f_x(y) \quad \text{s.t.} \quad g_x(y) \le 0, \qquad h_x(y) = 0,
```

with ``h_x : \mathbb{R}^n \to \mathbb{R}^{n - m}`` (the problem is not
overdetermined) and ``g_x : \mathbb{R}^n \to \mathbb{R}^{n_\text{ineq}}``.

**Representation.** The package reads every constraint of the JuMP model as a
row ``\ell_i \le F_i(y, x) \le u_i`` in the `L2OCore.dual_layout` order (a
[`AbstractProblemForm`](@ref)). Rows with ``\ell_i = u_i`` are the equalities,
``h_{x,i}(y) = F_i(y, x) - \ell_i``. The remaining rows with a finite bound are
the inequalities. For an inequality row, the signed distance to its interval,

```math
r_i(y) = F_i(y, x) - \Pi_{[\ell_i, u_i]}\big(F_i(y, x)\big),
```

satisfies ``r_i^2 = \mathrm{ReLU}(g_{x,i}(y))^2`` for one-sided rows and
handles two-sided rows (``\ell_i \le F_i \le u_i``) as one row. Its gradient
in ``F`` is ``\nabla_F \tfrac12 \|r\|^2 = r``, the projection residual. Every
violation and every correction direction in the package is computed from this
one quantity ([`inequality_residual`](@ref), [`total_violation`](@ref)).
Variable bounds are rows like any other.

Maximization problems are handled by minimizing ``s \cdot f`` with
``s = `` [`objective_sign`](@ref).

## DC3

### Equality completion (Section 3.1 of [DC3])

Split ``y = (z, \varphi_x(z))``, where ``z \in \mathbb{R}^m`` are the
*independent* variables predicted by the network and ``\varphi_x(z)`` are the
*dependent* variables defined by

```math
h_x\big([z;\ \varphi_x(z)]\big) = 0 .
```

Differentiating this identity gives the implicit-function-theorem Jacobian
(eq. (3) of [DC3])

```math
\frac{\partial \varphi_x(z)}{\partial z} = -\big(J^h_{:,\varphi}\big)^{-1} J^h_{:,z},
```

and backpropagation needs only the vector-Jacobian product (eq. (4)):

```math
\bar z = \bar y_z - \big(J^h_{:,z}\big)^{\top} w, \qquad \big(J^h_{:,\varphi}\big)^{\top} w = \bar y_\varphi .
```

The same argument gives ``\bar x = -(J^h_{:,x})^\top w`` for the parameters,
which [`NewtonCompletion`](@ref) also returns.

**Affine equalities** ``A_E y + B_E x + b_E = \ell_E``
([`LinearCompletion`](@ref)): with ``D`` the dependent and ``Z`` the
independent columns,

```math
y_D = -A_{E,D}^{-1}\big(A_{E,Z} z + B_E x + b_E - \ell_E\big),
```

which is precomputed as one affine map ``Y = K_z Z + K_x X + k_0``. Completion
is then a single batched matrix product, and its Jacobian ``K_z`` is exact.

**Nonlinear equalities** ([`NewtonCompletion`](@ref)): damped Newton's method
on ``y_D`` for each instance, with the derivatives above.

**Choice of the dependent variables.** Any ``D`` with ``A_{E,D}`` (or
``J^h_{:,D}`` at a reference point) nonsingular is valid. [DC3] draws random
subsets until the block is nonsingular. Here [`select_dependent`](@ref) uses
QR with column pivoting, first on ``A_E^\top`` to drop redundant equality rows
and then on the kept rows to pick well-conditioned columns. Dropped rows are
checked to be consistent for every parameter value: ``[A_E\ B_E\ b_E]``
restricted to them must lie in the row span of the kept rows. Otherwise the
problem is infeasible for some ``x`` and an error is thrown.

### Inequality correction (Section 3.2 of [DC3])

With the reduced inequality violation

```math
v(z) = \big\| \mathrm{ReLU}\big(g_x([z;\ \varphi_x(z)])\big) \big\|_2^2 ,
```

one correction step is ``z \leftarrow z - \gamma \Delta z`` with

```math
\Delta z = \nabla_z v(z) = \Big(\frac{\partial y}{\partial z}\Big)^{\!\top} J^g(y)^\top\, 2\,\mathrm{ReLU}(g_x(y)),
\qquad y = [z;\ \varphi_x(z)],
```

followed by re-completion ([`reduced_violation_gradient`](@ref),
[`inequality_correction`](@ref)). As in the reference implementation (momentum
``0.5``, Appendix B of [DC3]), the steps use heavy-ball momentum,
``v_{k+1} = \mu v_k + \Delta z_k`` and ``z_{k+1} = z_k - \gamma v_{k+1}``,
which equals the reference update ``s \leftarrow \gamma \Delta y + \mu s``,
``y \leftarrow y - s``. ``\rho^{(t)}`` denotes ``t`` steps; training
uses ``t_\text{train}`` and testing ``t_\text{test}`` (Algorithm 1 of [DC3]).

For affine equalities, re-completion gives exactly the paper's step
``y - \gamma[\Delta z;\ (\partial\varphi/\partial z)\,\Delta z]``. For
nonlinear equalities the paper's step keeps the equalities only to first order;
re-completion keeps them to the Newton tolerance (see *Deviations* below).

For affine constraints ``\ell \le A y + B x + b \le u``, the reduced violation
is

```math
v(z) = \operatorname{dist}^2\big(\hat G z + \hat H x + \hat g_0,\ [\ell_I, u_I]\big),
\qquad \hat G = A_I K_z,
```

a convex function whose gradient ``2 \hat G^\top r`` is Lipschitz with
``L = 2 \|\hat G\|_2^2``. This holds because the projection residual is
1-Lipschitz (firm non-expansiveness of the projection). The default step is
``\gamma = (1 - \mu)/L`` ([`default_step_size`](@ref) times ``1 - \mu``). For
``\mu = 0`` this is ``1/L``, the step of the descent lemma. For ``\mu > 0`` it
lies in the range where the averaged heavy-ball iterates converge on convex
``L``-smooth functions (Ghadimi, Feyzmahdavian & Johansson, 2015). The paper
instead fixes ``\gamma = 10^{-7}`` for its QPs, "as most methods went unstable
for higher correction learning rates" (Appendix B.1 of [DC3]).

**On the paper's convergence statement.** Section 3.2 of [DC3] says that for
problems with linear constraints "the correction process is mathematically
guaranteed to converge". The paper gives no proposition or proof for this. The
claims `CorrectionDescent` and `CorrectionConvergesAffine` below make it
precise: the assumptions are a step ``\gamma \le 1/L`` and a nonempty
feasible set; convergence holds without momentum, and for averaged iterates
with momentum.

### Training loss (eq. (2) of [DC3])

```math
\ell(\hat y) = f_x(\hat y) + \lambda_g \big\|\mathrm{ReLU}(g_x(\hat y))\big\|_2^2 + \lambda_h \big\|h_x(\hat y)\big\|_2^2,
\qquad \hat y = \rho_x^{(t_\text{train})}\big([z;\ \varphi_x(z)]\big),\quad z = N_\theta(x),
```

averaged over the batch ([`soft_loss`](@ref)). The gradient flows through the
completion (eq. (4)) and through the unrolled corrections.

The paper's equation uses squared norms. The authors' reference code instead
uses plain norms ``\|\mathrm{ReLU}(g)\|_2`` and ``\|h\|_2``, with
``\lambda_g = \lambda_h = 5`` (total weight 10, equality fraction 0.5).
`DC3(; penalty = :norm, λ_ineq = 5, λ_eq = 5)` reproduces that loss; the default
is the paper's equation.

### Differentiating unrolled corrections

The correction direction is itself a gradient, so differentiating
``\hat y(\theta)`` needs second derivatives. The package never asks the AD
system for an inner gradient (no nested AD). Instead:

- **affine problems:** ``\Delta z = 2 \hat G^\top r`` is explicit array code,
  and any first-order AD differentiates through it;
- **nonlinear problems:** ``\Delta z`` is assembled from two primitives with
  ChainRules rules that use exact Hessian-of-the-Lagrangian products
  (`MathOptInterface.eval_hessian_lagrangian_product`):
  - ``G = J_y F(y)^\top W`` ([`constraint_vjp`](@ref)): ``\bar W = J_y F\, \bar G`` and
    ``\bar{(y, x)} = \big(\sum_i W_i \nabla^2 F_i\big) [\bar G; 0]``;
  - ``w = (J^h_{:,D})^{-\top} r``: ``\bar r = s := (J^h_{:,D})^{-1} \bar w`` and
    ``\bar{(y, x)} = -\big(\sum_i w_i \nabla^2 h_i\big)\, \hat s``, where ``\hat s`` is ``s`` placed on
    the dependent coordinates.

  With these, ``\Delta z = \big(G - J^{h\top} w\big)_Z`` for
  ``G = J^{g\top} 2r``, and ``w`` solves the transposed system with right-hand
  side ``G_D``.

All of these are checked against finite differences in the test suite.

## FSNet

FSNet predicts all variables ``y_\theta(x)`` and minimizes the total violation
(eq. (3) of [FSNet])

```math
\phi(s; x) = \|h_x(s)\|_2^2 + \|\mathrm{ReLU}(g_x(s))\|_2^2
           = \|F(s, x) - \Pi_{[\ell, u]}(F(s, x))\|_2^2
```

by gradient descent from the prediction (eq. (4)),
``s_0 = y_\theta(x)``, ``s_{k+1} = s_k - \eta\, \nabla_s \phi(s_k; x)``
([`feasibility_seeking`](@ref)), truncated to ``K`` iterations. Derivatives flow
through the first ``K' \le K`` iterations only. The remaining ones add a
constant offset (truncated unrolled differentiation, Section 5 of [FSNet]). The
training loss is (eqs. (5)–(6), with the optional early-stage penalty of
Section 5)

```math
F = f_x(\hat y) + \frac{\rho}{2}\|y_\theta(x) - \hat y\|_2^2
    + \rho_\phi\, \phi(y_\theta(x))\, [\phi(y_\theta(x)) \ge Q] .
```

For affine constraints, ``\phi`` is convex with ``L = 2\|A\|_2^2``
([`lipschitz_constant`](@ref)), and the default step is ``\eta = 1/L``.

## Claims and guarantees

Each claim has a name (a type), a precise statement with its assumptions, and a
numerical check (`check(claim, ...)`). The names are the anchors to which
formal proofs can be attached later.

| name | statement | assumptions | checked in |
|:--|:--|:--|:--|
| `EqualityFeasible` (L2OCore) | ``h_x(\hat y) = 0`` to tolerance for every output of `complete(pipe, ...)` (DC3) | affine ``h``: none beyond ``A_{E,D}`` nonsingular (enforced at construction); nonlinear ``h``: Newton converges (``h \in C^2``, ``J^h_{:,D}`` nonsingular near the root, start in the basin) | `test_completion.jl`, `test_correction.jl`, `test_methods.jl` |
| [`CompletionJacobian`](@ref) | the VJP used for ``\varphi_x`` equals the derivative of the completion map (eq. (3)–(4)) | ``h \in C^1`` near the completed point, ``J^h_{:,D}`` nonsingular | `test_completion.jl` |
| [`CorrectionDescent`](@ref) | ``v(z_{k+1}) \le v(z_k) - \tfrac{\gamma}{2}\|\nabla v(z_k)\|^2`` | ``\nabla v`` is ``L``-Lipschitz, ``0 < \gamma \le 1/L``, momentum ``0``. Holds for affine constraints with the default step. | `test_correction.jl` |
| `CorrectionConvergesAffine` (test name) | for affine constraints and a nonempty feasible set, ``v(z_k) \to 0`` (rate ``O(1/k)``, linear by Hoffman's bound) | as above, plus feasibility; with momentum ``\mu``, step ``(1-\mu)/L`` (averaged iterates) | `test_correction.jl`, `test_reference.jl` |
| reference equivalence | completion, reduced gradient (`G_eff`, `h_eff`), momentum steps and the loss equal the formulas of the authors' code | affine equalities | `test_reference.jl` |
| [`FeasibilitySeekingDescent`](@ref) | ``\phi(s_{k+1}) \le \phi(s_k) - \tfrac{\eta}{2}\|\nabla\phi(s_k)\|^2`` | ``\nabla\phi`` is ``L``-Lipschitz, ``0 < \eta \le 1/L`` | `test_correction.jl` |
| FSNet Theorem 1 | ``\phi(s_k) \le (1 - \mu/L)^k \phi(s_0)`` | ``\phi`` is ``L``-smooth and satisfies PL with constant ``\mu``, and a feasible point exists. Holds for affine constraints with a nonempty feasible set. | `test_correction.jl` (convergence to ``\phi < 10^{-12}``) |
| gradient correctness | Zygote gradients of every loss, completion, correction and form primitive equal central finite differences | smoothness at the test points | all test files |
| solver agreement | completion and correction reproduce the solver optimum from its independent part. At the optimum the soft loss equals ``f^\star``. Descent on the DC3 loss reaches feasible points within 1–2% of HiGHS (QP, LP) and returns to Ipopt's local optimum. | convexity (QP/LP); local analysis (NLP) | `test_optima.jl` |

**Proof sketches.**

- *CorrectionDescent / FeasibilitySeekingDescent.* This is the descent lemma:
  for ``L``-smooth ``v`` and ``\gamma \le 1/L``,
  ``v(z - \gamma\nabla v) \le v(z) - \gamma(1 - L\gamma/2)\|\nabla v\|^2 \le v(z) - \tfrac{\gamma}{2}\|\nabla v\|^2``.
  No convexity is needed. With momentum the iterates are not monotone in
  general, so the claim is stated for ``\mu = 0``.
- *Affine smoothness.* ``v(z) = \|\hat r(z)\|^2`` with
  ``\hat r = \hat F - \Pi(\hat F)`` and ``\hat F = \hat G z + c(x)``. The map
  ``F \mapsto F - \Pi(F)`` is 1-Lipschitz, so
  ``\|\nabla v(z) - \nabla v(z')\| = 2\|\hat G^\top(\hat r(z) - \hat r(z'))\| \le 2\|\hat G\|^2\|z - z'\|``.
- *CorrectionConvergesAffine and PL for affine constraints.* The function
  ``v`` is convex (a squared distance to a convex set, composed with an affine
  map) and ``L``-smooth. Gradient descent with ``\gamma = 1/L`` gives
  ``v(z_k) - v^\star \le L\|z_0 - z^\star\|^2 / (2k)``. If the reduced feasible
  set is nonempty, ``v^\star = 0``. By Hoffman's error bound for polyhedra,
  ``v(z) \ge c\, \operatorname{dist}(z, S)^2`` (quadratic growth), which for
  convex functions is equivalent to the PL inequality (Karimi, Nutini &
  Schmidt, 2016). This gives a linear rate, as in Theorem 1 of [FSNet]. The
  same argument applies to FSNet's ``\phi`` with ``A`` in place of ``\hat G``.
  The paper's instances are feasible by construction (eq. (6) of [DC3]).

## Deviations from the papers

- **Dependent-variable selection:** pivoted QR instead of random subsets
  (deterministic and better conditioned). Users can pass `dependent`.
- **Nonlinear correction:** re-completion after each ``z``-step instead of the
  first-order step on ``\varphi``, so the equalities hold to the Newton
  tolerance after every step. This is identical for affine equalities.
- **Second-order terms** through the nonlinear correction are exact, from
  MathOptInterface Hessian-of-the-Lagrangian products rather than autograd of
  hand-written Jacobians as in the reference code.
- **FSNet inner solver:** gradient descent, the setting of the paper's
  theory. The experiments in [FSNet] use L-BFGS with line search; this is a
  planned extension.
- **FSNet truncation:** derivatives through the first ``K'`` of ``K`` iterations
  (as stated in Section 5 of [FSNet]), with the later iterations entering as a
  constant offset.
- **Jacobian formed explicitly for affine equalities:** [DC3] recommends
  vector-Jacobian products instead of forming ``\partial\varphi/\partial z``.
  For affine equalities the Jacobian is constant, so ``K_z`` is formed once and
  completion becomes one batched matrix product (GPU-friendly). Nonlinear
  completion uses VJPs as recommended.
- **Newton completion** is damped (backtracking on ``\|h\|_\infty``). Appendix
  C.3 of [DC3] reports that solving for all variables with Newton's method "is
  sometimes unstable" on AC-OPF and splits the completion into a Newton part
  and a closed-form part. Such problem-specific completions plug in through
  [`AbstractCompletion`](@ref).
- **Defaults and their provenance:**

  | | paper | authors' code | this package |
  |:--|:--|:--|:--|
  | soft loss | squared norms, eq. (2) | plain norms | squared (`penalty = :norm` for the code's form) |
  | ``\lambda_g, \lambda_h`` | sum tuned in ``\{1, 10, 100\}`` | 5, 5 | 10, 10 (as in the FSNet authors' re-implementation of DC3) |
  | ``t_\text{train}, t_\text{test}`` | 10, 10 (QP) | 10, 10 | 10, 10 |
  | correction step ``\gamma`` | ``10^{-7}`` (QP) | ``10^{-7}`` (QP), ``10^{-4}`` (AC-OPF) | ``(1-\mu)/L`` (affine); required (nonlinear) |
  | momentum ``\mu`` | 0.5 | 0.5 | 0.5 |
  | test-time stopping tolerance | ``10^{-4}`` | ``10^{-4}`` (equalities and inequalities) | `test_tol = 0`: exactly ``t_\text{test}`` steps (Algorithm 1) |
  | dependent variables | not specified | random subset with ``\lvert\det A_D\rvert \ge 10^{-4}`` | pivoted QR, or `dependent` |
  | FSNet ``\rho`` (distance weight ``\rho/2``) | tuned | ``\rho/2 = 5`` | ``\rho = 10`` |
  | FSNet early penalty ``\rho_\phi, Q`` | per instance, Section 5 | 10, ``10^3``, on batch means | 10, ``10^3``, per instance |
  | FSNet inner solver | gradient descent (theory), L-BFGS (experiments) | L-BFGS, 50 iterations, 30 differentiated | gradient descent |
