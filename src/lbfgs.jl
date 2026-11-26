# Batched L-BFGS for FSNet's feasibility problem min_s φ(s) = ‖F(s) - Π_C(F(s))‖²
# (the inner solver of the experiments of Nguyen & Donti, 2025, Section 6).
# Every column is an independent problem: curvature pairs, two-loop recursion
# and Armijo backtracking are all column-wise.

_coldot(a, b) = sum(a .* b; dims=1)

# Two-loop recursion: r ≈ H_k g for each column (Nocedal & Wright, Alg. 7.4).
# The history is a tuple (immutable, so reverse-mode AD can follow it).
function _lbfgs_direction(G, Ss::Tuple, Ys::Tuple, ρs::Tuple)
    k = length(Ss)
    k == 0 && return -G
    q = G
    αs = ()
    for i in k:-1:1
        α = ρs[i] .* _coldot(Ss[i], q)
        q = q .- α .* Ys[i]
        αs = (α, αs...)
    end
    yy = _coldot(Ys[k], Ys[k])
    sy = _coldot(Ss[k], Ys[k])
    γ = ifelse.(yy .> 0, sy ./ max.(yy, eps(eltype(G))), one(eltype(G)))
    r = γ .* q
    for i in 1:k
        β = ρs[i] .* _coldot(Ys[i], r)
        r = r .+ Ss[i] .* (αs[i] .- β)
    end
    return -r
end

# Armijo backtracking per column, outside of AD: returns step lengths (1 × B).
function _armijo(form, S, P, φ, G, D, t0; c=1e-4, shrink=0.5, maxiter=30)
    slope = _coldot(G, D)
    t = fill!(similar(φ), t0)
    for _ in 1:maxiter
        φt = reshape(total_violation(form, S .+ t .* D, P), 1, :)
        ok = φt .<= φ .+ c .* t .* slope
        all(ok) && break
        t = ifelse.(ok, t, shrink .* t)
    end
    return t
end

struct _LBFGSState{A<:Tuple,B<:Tuple,C<:Tuple}
    Ss::A
    Ys::B
    ρs::C
end

_push(h::Tuple, x, memory) = length(h) >= memory ? (Base.tail(h)..., x) : (h..., x)

function _lbfgs_iterations(form, S, P, state, iters, memory, step0, tol)
    Ss, Ys, ρs = state.Ss, state.Ys, state.ρs
    G = violation_gradient(form, S, P)
    for _ in 1:iters
        φ = _nograd(() -> reshape(total_violation(form, S, P), 1, :))
        if tol > 0 && _nograd(() -> maximum(φ)) <= tol
            break
        end
        D = _lbfgs_direction(G, Ss, Ys, ρs)
        # Fall back to steepest descent where the direction is not a descent one.
        D =
            _nograd(() -> _coldot(G, D) .< 0) .* D .-
            _nograd(() -> _coldot(G, D) .>= 0) .* G
        t0 = isempty(Ss) ? step0 : one(step0)
        t = _nograd(() -> _armijo(form, S, P, φ, G, D, t0))
        Snew = S .+ t .* D
        Gnew = violation_gradient(form, Snew, P)
        s, y = Snew .- S, Gnew .- G
        sy = _coldot(s, y)
        # Keep a pair only where the curvature condition holds (ρ = 0 skips it).
        ok = _nograd(() -> sy .> eps(eltype(sy)))
        ρ = ok ./ max.(sy, eps(eltype(sy)))   # differentiable, as in the reference
        Ss, Ys, ρs = _push(Ss, s, memory), _push(Ys, y, memory), _push(ρs, ρ, memory)
        S, G = Snew, Gnew
    end
    return S, _LBFGSState(Ss, Ys, ρs)
end

"""
    feasibility_seeking_lbfgs(form, Y, P; steps, unroll = steps, memory = 10,
                              step_size = 1, tol = 0) -> Ŷ

FSNet's feasibility-seeking map computed with L-BFGS, the inner solver used in
the experiments of Nguyen & Donti (2025): `steps` iterations of the two-loop
recursion with `memory` curvature pairs and Armijo backtracking on
`φ(s) = ‖h(s)‖² + ‖ReLU(g(s))‖²`, for every column independently. The first
iteration starts from `step_size` (e.g. `1/L`), later ones from a unit step.

Derivatives flow through the first `unroll` iterations; the line-search step
lengths are treated as constants (as in the authors' implementation), and the
remaining iterations add a constant offset (truncated unrolled
differentiation).

Armijo backtracking makes `φ` non-increasing along the iterates.
"""
function feasibility_seeking_lbfgs(
    form::AbstractProblemForm,
    Y::AbstractMatrix,
    P::AbstractMatrix;
    steps::Integer,
    unroll::Integer=steps,
    memory::Integer=10,
    step_size::Real=1,
    tol::Real=0,
)
    0 <= unroll || throw(ArgumentError("unroll must be nonnegative"))
    memory >= 1 || throw(ArgumentError("memory must be at least 1"))
    T = eltype(Y)
    empty = _LBFGSState((), (), ())
    K = min(unroll, steps)
    S, state = _lbfgs_iterations(form, Y, P, empty, K, memory, convert(T, step_size), tol)
    K == steps && return S
    S0 = _stopgrad(S)
    Tail = _nograd() do
        first(
            _lbfgs_iterations(
                form, S0, P, state, steps - K, memory, convert(T, step_size), tol
            ),
        )
    end
    return S .+ (Tail .- S0)
end
