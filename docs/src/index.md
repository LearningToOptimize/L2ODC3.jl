# L2ODC3.jl

L2ODC3.jl trains neural-network proxies for parametric optimization problems
that must satisfy hard constraints. It implements two methods on the
`L2OCore` interface of the
[LearningToOptimize](https://github.com/LearningToOptimize) ecosystem:

- [`DC3`](@ref), *Deep Constraint Completion and Correction*
  ([Donti, Rolnick & Kolter, ICLR 2021](https://arxiv.org/abs/2104.12225)). The
  proxy predicts a subset of the variables, and the equality constraints are
  solved for the rest (exactly for affine equalities, by Newton's method
  otherwise), so every prediction satisfies them. Unrolled gradient steps
  along the equality manifold then remove the inequality violation. Training
  is self-supervised with a soft penalty loss.
- [`FSNet`](@ref), *Feasibility-Seeking Neural Network*
  ([Nguyen & Donti, 2025](https://arxiv.org/abs/2506.00362)). The proxy
  predicts every variable, and unrolled gradient descent on the total violation
  ``\|h\|^2 + \|\mathrm{ReLU}(g)\|^2`` moves the prediction to a feasible point.

Problems are JuMP models with parameters. The core is framework-agnostic: any
callable proxy and any ChainRules-compatible AD (Zygote is tested). Affine
problems run on GPU arrays.

```mermaid
flowchart LR
    x["parameters x"] --> NN["proxy N_θ"]
    NN --> z["independent z"]
    z --> phi["completion φ_x(z): h_x(y) = 0"]
    phi --> rho["correction ρ_x: steps on ReLU(g_x)²"]
    rho --> y["ŷ"]
    y --> loss["f_x(ŷ) + λg‖ReLU(g_x(ŷ))‖² + λh‖h_x(ŷ)‖²"]
    loss -. "backprop through ρ and φ (implicit function theorem)" .-> NN
```

- [Theory](theory.md): the methods as optimization problems, their
  derivatives and step sizes, and where the package departs from the papers.
- [Package guide](guide.md): getting started, concepts, GPU, AD support and
  extension points.
- [Example](examples/dc3_qp.md): training a DC3 and an FSNet proxy on the
  convex QP of the DC3 paper and comparing with HiGHS.
- [API reference](api.md).

## Citing

If you use DC3 or FSNet, cite the original papers (see `CITATION.cff` in the
repository for this package):

```bibtex
@inproceedings{donti2021dc3,
  title     = {{DC3}: A learning method for optimization with hard constraints},
  author    = {Donti, Priya L. and Rolnick, David and Kolter, J. Zico},
  booktitle = {International Conference on Learning Representations},
  year      = {2021},
}
@article{nguyen2025fsnet,
  title   = {{FSNet}: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees},
  author  = {Nguyen, Hoang T. and Donti, Priya L.},
  journal = {arXiv preprint arXiv:2506.00362},
  year    = {2025},
}
```
