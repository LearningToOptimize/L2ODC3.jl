# Numerical checks of the properties derived on the Theory page ("Convergence of
# the corrections"). Each function returns one error per instance; the tests
# compare the largest error with a tolerance.

"""
Largest equality violation of `pred` per instance, from `L2OCore.evaluate`.
"""
function max_equality_violation(problem, P, pred; kwargs...)
    return maximum(evaluate(problem, P, pred; kwargs...).max_equality_violation)
end

"""
Largest violation of any constraint of `pred`, from `L2OCore.evaluate`.
"""
function max_violation(problem, P, pred; kwargs...)
    return maximum(evaluate(problem, P, pred; kwargs...).max_violation)
end

colsumsq(G) = vec(sum(abs2, G; dims=1))

"""
Relative error, per instance, between `vᵀ (∂φ/∂z) u` computed with
`completion_pullback` (implicit function theorem, eqs. (3)–(4) of Donti et al.)
and the central finite difference `vᵀ (φ(z + εu) - φ(z - εu)) / 2ε`, for random
directions `u` and `v`.
"""
function completion_jacobian_error(pipe, P, Z; rng, ε=1e-6)
    c = pipe.decomposition.completion
    P, Z = Matrix{Float64}(P), Matrix{Float64}(Z)
    Y = equality_completion(c, Z, P)
    U = randn(rng, size(Z)...)
    V = randn(rng, size(Y)...)
    Yp = equality_completion(c, Z .+ ε .* U, P)
    Ym = equality_completion(c, Z .- ε .* U, P)
    fd = vec(sum(V .* (Yp .- Ym); dims=1)) ./ (2ε)
    an = vec(sum(U .* completion_pullback(c, Y, P, V); dims=1))
    return abs.(fd .- an) ./ max.(1, abs.(an))
end

"""
Largest relative violation, per instance, of the sufficient-decrease inequality
`v(z_{k+1}) ≤ v(z_k) - (γ/2)‖∇v(z_k)‖²` over `steps` DC3 corrections without
momentum, with the pipeline's step size `γ`.
"""
function correction_descent_error(pipe, P, Z; steps=20)
    d = pipe.decomposition
    γ = d.step_size
    Y = equality_completion(d.completion, Z, P)
    err = zeros(size(Z, 2))
    for _ in 1:steps
        v = inequality_violation(d.form, Y, P)
        G = reduced_violation_gradient(d.form, d.completion, Y, P)
        Z = Z .- γ .* G
        Y = equality_completion(d.completion, Z, P; init=Y)
        vnew = inequality_violation(d.form, Y, P)
        bound = v .- (γ / 2) .* colsumsq(G)
        err .= max.(err, max.(vnew .- bound, 0) ./ max.(1, v))
    end
    return err
end

"""
Largest relative violation, per instance, of the sufficient-decrease inequality
`φ(s_{k+1}) ≤ φ(s_k) - (η/2)‖∇φ(s_k)‖²` over `steps` FSNet gradient steps with
the pipeline's step size `η`.
"""
function feasibility_descent_error(pipe, P, S; steps=20)
    d = pipe.decomposition
    η = d.step_size
    err = zeros(size(S, 2))
    for _ in 1:steps
        φ = total_violation(d.form, S, P)
        G = violation_gradient(d.form, S, P)
        S = S .- η .* G
        φnew = total_violation(d.form, S, P)
        bound = φ .- (η / 2) .* colsumsq(G)
        err .= max.(err, max.(φnew .- bound, 0) ./ max.(1, φ))
    end
    return err
end
