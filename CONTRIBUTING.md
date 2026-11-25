# Contributing

Contributions are welcome: bug reports, new completions or problem forms,
examples and documentation fixes.

- **Issues.** Describe the problem with a minimal JuMP model and the versions of
  Julia, L2OCore and this package.
- **Pull requests.** Keep them focused (one concern per commit). Add tests and
  documentation for every new feature.
- **Tests.** `julia --project -e 'using Pkg; Pkg.test()'`. Every mathematical
  claim must have a named, numerically checked test (see the Theory page):
  feasibility to tolerance, derivatives against finite differences,
  comparisons with a solver optimum. Use seeded RNGs (`StableRNGs`) and keep
  the suite fast.
- **Style.** BlueStyle via JuliaFormatter:
  `using JuliaFormatter; format(".")` (checked on CI).
- **Docs.** `julia --project=docs docs/make.jl`. Doctests run, and every
  exported symbol needs a docstring listed in `docs/src/api.md`.
- **Extending.** New completions subtype `AbstractCompletion`, new evaluation
  backends subtype `AbstractProblemForm`, and new claims subtype
  `L2OCore.AbstractGuarantee` with a `check` method. See the Package guide.
