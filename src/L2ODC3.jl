"""
    L2ODC3

Learning-to-optimize methods that enforce hard constraints by completion and
correction, built on `L2OCore`:

  - [`DC3`](@ref): Deep Constraint Completion and Correction (Donti, Rolnick &
    Kolter, ICLR 2021). The proxy predicts the independent variables, an
    equality completion recovers the rest so that the equality constraints hold
    exactly, and unrolled gradient steps reduce the inequality violation.
  - [`FSNet`](@ref): Feasibility-Seeking Neural Network (Nguyen & Donti, 2025).
    The proxy predicts every variable and unrolled gradient descent on the
    total violation `‖h‖² + ‖ReLU(g)‖²` seeks a feasible point.

Problems are JuMP models with parameters wrapped as
`L2OCore.ParametricProblem`; the package is framework-agnostic (any callable
proxy, any ChainRules-compatible AD).
"""
module L2ODC3

using ChainRulesCore: ChainRulesCore
using JuMP: JuMP
using L2OCore: L2OCore
using LinearAlgebra: LinearAlgebra
import MathOptInterface as MOI
using Random: Random

include("forms.jl")
include("completion.jl")
include("correction.jl")
include("methods.jl")
include("claims.jl")

# Methods and decompositions
export DC3, FSNet, DC3Decomposition, FSNetDecomposition
# Problem forms
export AbstractProblemForm,
    AffineForm,
    NLPForm,
    problem_form,
    row_bounds,
    equality_indices,
    inequality_indices,
    objective_sign,
    constraint_vjp,
    constraint_jacobian,
    min_objective,
    equality_residual,
    inequality_residual,
    equality_violation,
    inequality_violation,
    total_violation,
    lipschitz_constant
# Completion
export AbstractCompletion,
    LinearCompletion,
    NewtonCompletion,
    select_dependent,
    independent_indices,
    dependent_indices,
    equality_completion,
    completion_pullback
# Correction and feasibility seeking
export reduced_violation_gradient,
    inequality_correction, violation_gradient, feasibility_seeking, default_step_size
# Losses and batch entry points
export soft_loss, predict, training_loss
# Named claims
export CompletionJacobian, CorrectionDescent, FeasibilitySeekingDescent

end # module L2ODC3
