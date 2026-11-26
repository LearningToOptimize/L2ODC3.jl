# Theory

This page states the two methods as optimization problems, fixes the notation,
derives the derivatives and step sizes the implementation uses, and records
where the package departs from the papers. Equation numbers in parentheses
refer to the papers:

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

**On the paper's convergence statement.** Section 3.2 of [DC3] states that,
for problems with linear constraints, the correction process converges, without
a precise statement. [Convergence of the corrections](@ref) below gives one:
with a step ``\gamma \le 1/L`` and a nonempty feasible set, the violation goes
to zero without momentum, and the averaged iterates converge with momentum.

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

## Second-order cones

Conic inequalities enter through the same projection residual. For any
closed convex set ``C``, the function ``\tfrac12 \operatorname{dist}^2(F, C)``
is differentiable, with gradient ``F - \Pi_C(F)``, and the map
``F \mapsto F - \Pi_C(F)`` is 1-Lipschitz (Moreau). The package uses this for
`MOI.SecondOrderCone` blocks
``K = \{(t, x) : \|x\|_2 \le t\}``, whose projection has the closed form

```math
\Pi_K(t, x) =
\begin{cases}
(t, x) & \|x\| \le t,\\
0 & \|x\| \le -t,\\
\tfrac{t + \|x\|}{2}\,\big(1,\ x/\|x\|\big) & \text{otherwise},
\end{cases}
```

([`soc_residual`](@ref)). `MOI.RotatedSecondOrderCone` blocks
``\{(t, u, x) : 2tu \ge \|x\|^2,\ t, u \ge 0\}`` are mapped to it by the
orthogonal map ``(t, u) \mapsto ((t + u)/\sqrt2,\ (t - u)/\sqrt2)``
([`rsoc_residual`](@ref)). [`projection_residual`](@ref) stacks the interval
residuals of box rows and the cone residuals of cone blocks in row order.
Consequently:

- the inequality violation ``v`` of DC3 and FSNet's ``\phi`` include
  ``\operatorname{dist}^2(F_K, K)`` for every cone block;
- the correction directions are ``J^\top 2(F - \Pi_C(F))``, unchanged in form;
- the Lipschitz constants ``L = 2\|A_I K_z\|_2^2`` and ``L = 2\|A\|_2^2``
  still hold (with ``I`` including the cone rows), because the residual is
  1-Lipschitz. The descent inequalities and the convergence arguments of
  [Convergence of the corrections](@ref) carry over unchanged: ``v`` is still
  a convex, ``L``-smooth squared distance. The linear rate via Hoffman's bound
  does not carry over, since that bound is specific to polyhedra; the
  ``O(1/k)`` rate does.

The tests verify the Moreau decomposition (``\Pi(F) \in K``,
``-r \in K^\ast = K``, ``\langle \Pi(F), r \rangle = 0``),
``\nabla\operatorname{dist}^2 = 2r`` by AD, agreement with MathOptInterface
distances, monotone descent on a portfolio SOCP, and agreement with the
Clarabel optimum (`test_cones.jl`). The norm is
smoothed as ``\sqrt{\|x\|^2 + \varepsilon^2}`` with machine ``\varepsilon`` so
that derivatives are finite at ``x = 0``.

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

**L-BFGS** (`FSNet(; solver = :lbfgs)`, [`feasibility_seeking_lbfgs`](@ref)) is
the inner solver of the paper's experiments. Each instance (column) is an
independent problem: the two-loop recursion (Nocedal & Wright, Algorithm 7.4)
uses that column's curvature pairs ``(s_i, y_i)``. A pair is dropped where
``s_i^\top y_i \le \varepsilon``. A non-descent direction falls back to
``-\nabla\phi``, and Armijo backtracking
``\phi(s + t d) \le \phi(s) + c\, t\, \nabla\phi^\top d`` makes ``\phi``
non-increasing. As in the authors' code, derivatives flow through the
iterates, the curvature pairs and the two-loop recursion, while the
line-search step lengths are held constant (they are piecewise constant in the
input). Truncation to the first ``K'`` iterations is as for gradient descent.

## Convergence of the corrections

**Completion.** If ``A_{E,D}`` is nonsingular (the dependent-variable selection
above enforces it), the affine completion satisfies ``h_x(\hat y) = 0`` up to
round-off for every proxy output (Section 3.1 of [DC3]). For nonlinear ``h``,
the completion satisfies the equalities to the Newton tolerance whenever
Newton's method converges, which is the case for ``h \in C^2`` with
``J^h_{:,D}`` nonsingular near the root and a start in its basin. Under the same
nonsingularity, and ``h \in C^1`` near the completed point, the implicit
function theorem gives eqs. (3)–(4), so the vector-Jacobian product above is
the derivative of the completion map.

**Descent.** If ``\nabla v`` is ``L``-Lipschitz and ``0 < \gamma \le 1/L``, a
correction step without momentum decreases the violation by the descent lemma,

```math
v(z - \gamma\nabla v(z)) \le v(z) - \gamma\big(1 - \tfrac{L\gamma}{2}\big)\|\nabla v(z)\|^2
\le v(z) - \tfrac{\gamma}{2}\|\nabla v(z)\|^2 .
```

No convexity is needed. The same inequality holds for FSNet's
``\phi(s_{k+1}) \le \phi(s_k) - \tfrac{\eta}{2}\|\nabla\phi(s_k)\|^2`` with
``0 < \eta \le 1/L``. With momentum the iterates are not monotone in general.

**Smoothness for affine constraints.** Write ``v(z) = \|\hat r(z)\|^2`` with
``\hat r = \hat F - \Pi(\hat F)`` and ``\hat F = \hat G z + c(x)``. The map
``F \mapsto F - \Pi(F)`` is 1-Lipschitz, so

```math
\|\nabla v(z) - \nabla v(z')\| = 2\|\hat G^\top(\hat r(z) - \hat r(z'))\| \le 2\|\hat G\|^2\|z - z'\| ,
```

which is the constant ``L`` used for the default step.

**Rates for affine constraints.** The function ``v`` is convex (a squared
distance to a convex set, composed with an affine map) and ``L``-smooth.
Gradient descent with ``\gamma = 1/L`` gives
``v(z_k) - v^\star \le L\|z_0 - z^\star\|^2 / (2k)``. If the reduced feasible
set ``S`` is nonempty, ``v^\star = 0``. By Hoffman's error bound for polyhedra,
``v(z) \ge c\, \operatorname{dist}(z, S)^2`` (quadratic growth), which for
convex functions is equivalent to the Polyak–Łojasiewicz inequality (Karimi,
Nutini & Schmidt, 2016). With PL constant ``\kappa`` this gives the linear rate
``v(z_k) \le (1 - \kappa/L)^k v(z_0)``. The same argument applies to FSNet's
``\phi`` with ``A`` in place of ``\hat G``, and gives
``\phi(s_k) \le (1 - \kappa/L)^k \phi(s_0)``, the rate of Theorem 1 of [FSNet]
(where the PL constant is called ``\mu``). With
momentum ``\mu`` and step ``(1 - \mu)/L``, the averaged heavy-ball iterates
converge at rate ``O(1/k)`` (Ghadimi, Feyzmahdavian & Johansson, 2015). The
paper's instances are feasible by construction (eq. (6) of [DC3]).

**Agreement with the reference code and with solvers.** For affine equalities,
the completion, the reduced gradient (`G_eff`, `h_eff`), the momentum steps
and the loss coincide with the formulas of the authors' code. Starting from the
independent part of a solver optimum, completion and correction reproduce that
optimum, and the soft loss there equals ``f^\star``. Descent on the DC3 loss
reaches feasible points within 1–2% of HiGHS on QPs and LPs, and returns to
Ipopt's local optimum on a nonlinear problem.

## Deviations from the papers

- **Dependent-variable selection:** pivoted QR instead of random subsets
  (deterministic and better conditioned). Users can pass `dependent`.
- **Nonlinear correction:** re-completion after each ``z``-step instead of the
  first-order step on ``\varphi``, so the equalities hold to the Newton
  tolerance after every step. This is identical for affine equalities.
- **Second-order terms** through the nonlinear correction are exact, from
  MathOptInterface Hessian-of-the-Lagrangian products rather than autograd of
  hand-written Jacobians as in the reference code.
- **FSNet inner solver:** gradient descent by default, the setting of the
  paper's theory. L-BFGS with Armijo backtracking, the solver of the
  experiments, is available as `solver = :lbfgs`. It is per instance, whereas
  the authors' code runs L-BFGS on the batch-mean violation.
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
- **Defaults compared with the paper and the authors' code:**

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
  | FSNet inner solver | gradient descent (theory), L-BFGS (experiments) | L-BFGS, 50 iterations, 30 differentiated | gradient descent (`solver = :gd`) or L-BFGS (`solver = :lbfgs`) |
