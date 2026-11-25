# Experiments

Exploratory scripts that are not part of the test suite (they pull heavy AD
backends and record which combinations work rather than asserting a contract).

- `di_over_optimisers.jl`: nested differentiation (outer AD over an unrolled
  `Optimisers.Adam` loop whose inner gradients come from DifferentiationInterface)
  for every pair of ForwardDiff, Zygote, Enzyme and Mooncake.

Run with

```julia
julia --project=test/experiments -e 'using Pkg; Pkg.instantiate(); include("test/experiments/di_over_optimisers.jl")'
```

The DC3 and FSNet implementations in `src/` do not need nested AD: the
correction direction is computed in closed form (affine problems) or by
ChainRules rules with exact second derivatives (nonlinear problems), so the
training loss is differentiated by a single first-order reverse pass.
