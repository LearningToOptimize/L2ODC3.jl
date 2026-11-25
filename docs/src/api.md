# API reference

```@docs
L2ODC3
```

## Methods

```@docs
DC3
FSNet
DC3Decomposition
FSNetDecomposition
predict
training_loss
soft_loss
```

## Problem forms

```@docs
AbstractProblemForm
AffineForm
NLPForm
problem_form
row_bounds
equality_indices
inequality_indices
L2OCore.objective_sign(::AbstractProblemForm)
constraint_vjp
constraint_jacobian
min_objective
equality_residual
inequality_residual
equality_violation
inequality_violation
total_violation
lipschitz_constant
```

## Equality completion

```@docs
AbstractCompletion
LinearCompletion
NewtonCompletion
select_dependent
independent_indices
dependent_indices
equality_completion
completion_pullback
```

## Correction and feasibility seeking

```@docs
reduced_violation_gradient
inequality_correction
violation_gradient
feasibility_seeking
default_step_size
```

## Other exported names

```@docs
CompletionJacobian
CorrectionDescent
FeasibilitySeekingDescent
```

## L2OCore pipeline steps implemented here

```@docs
L2OCore.generate_data(::Union{L2ODC3.DC3,L2ODC3.FSNet}, ::L2OCore.L2OPipeline)
```
