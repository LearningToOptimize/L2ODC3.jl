# Second-order cone constraints through their projection residual.
#
# For any closed convex set C, d(F) = ½ dist²(F, C) is differentiable with
# ∇d(F) = F - Π_C(F) (Moreau), and F ↦ F - Π_C(F) is 1-Lipschitz. Box rows use
# Π = clamp; second-order cones have a closed-form projection. Every violation
# and correction direction of DC3 and FSNet is built from this one residual.

const _CONE_KINDS = (:soc, :rsoc)

_cone_kind(::MOI.SecondOrderCone) = :soc
_cone_kind(::MOI.RotatedSecondOrderCone) = :rsoc
function _cone_kind(s)
    throw(
        ArgumentError(
            "constraint set $(nameof(typeof(s))) is not supported; DC3/FSNet support box-like " *
            "sets, SecondOrderCone and RotatedSecondOrderCone",
        ),
    )
end

# Partition 1:m into maximal box ranges and the cone blocks, in row order.
function _segments(m::Int, cones::Vector{Tuple{UnitRange{Int},Symbol}})
    segs = Tuple{UnitRange{Int},Symbol}[]
    next = 1
    for (rows, kind) in sort(cones; by=first ∘ first)
        first(rows) > next && push!(segs, (next:(first(rows) - 1), :box))
        push!(segs, (rows, kind))
        next = last(rows) + 1
    end
    next <= m && push!(segs, (next:m, :box))
    return segs
end

function _cone_rows(segments)
    return reduce(vcat, [collect(r) for (r, k) in segments if k !== :box]; init=Int[])
end

"""
    cone_blocks(form::AbstractProblemForm) -> Vector{Tuple{UnitRange{Int},Symbol}}

Rows of each conic constraint and its kind (`:soc` for
`MOI.SecondOrderCone`, `:rsoc` for `MOI.RotatedSecondOrderCone`). Box-like rows
are not listed. The fallback is an empty vector.
"""
cone_blocks(::AbstractProblemForm) = Tuple{UnitRange{Int},Symbol}[]

"""
    violation_indices(form::AbstractProblemForm) -> AbstractVector{Int}

All inequality rows: the box inequalities ([`inequality_indices`](@ref)) and the
rows of every cone block, sorted. These are the rows of
[`inequality_residual`](@ref). The fallback is `inequality_indices(form)`.
"""
violation_indices(form::AbstractProblemForm) = inequality_indices(form)

"""
    soc_residual(F::AbstractMatrix) -> AbstractMatrix

`F - Π_K(F)` for every column of `F = [t; x]` and the second-order cone
`K = {(t, x) : ‖x‖₂ ≤ t}`. With `α = (t + ‖x‖)/2`, the projection is `F` if
`‖x‖ ≤ t`, `0` if `‖x‖ ≤ -t`, and `α (1, x/‖x‖)` otherwise. Its squared norm is
`dist²(F, K)` and `2(F - Π_K(F))` is the gradient of `dist²(·, K)`.

Branch-free broadcasts (GPU-friendly). `‖x‖` is smoothed as
`sqrt(‖x‖² + eps²)`, so the derivative is finite at `x = 0`.
"""
function soc_residual(F::AbstractMatrix)
    T = eltype(F)
    t = F[1:1, :]
    x = F[2:end, :]
    nx = sqrt.(sum(abs2, x; dims=1) .+ eps(float(real(T)))^2)
    inside = _nograd(() -> nx .<= t)
    polar = _nograd(() -> nx .<= -t)
    mid = _nograd(() -> .!(inside .| polar))
    α = (t .+ nx) ./ 2
    rt = polar .* t .+ mid .* (t .- α)
    rx = polar .* x .+ mid .* (x .- (α ./ nx) .* x)
    return vcat(rt, rx)
end

"""
    rsoc_residual(F::AbstractMatrix) -> AbstractMatrix

`F - Π_K(F)` for the rotated second-order cone
`K = {(t, u, x) : 2tu ≥ ‖x‖², t, u ≥ 0}`, through the orthogonal map
`(t, u) ↦ ((t + u)/√2, (t - u)/√2)` that sends it to the second-order cone.
"""
function rsoc_residual(F::AbstractMatrix)
    s = sqrt(convert(float(real(eltype(F))), 2))
    t, u = F[1:1, :], F[2:2, :]
    R = soc_residual(vcat((t .+ u) ./ s, (t .- u) ./ s, F[3:end, :]))
    r0, r1 = R[1:1, :], R[2:2, :]
    return vcat((r0 .+ r1) ./ s, (r0 .- r1) ./ s, R[3:end, :])
end

_cone_residual(kind::Symbol, F) = kind === :soc ? soc_residual(F) : rsoc_residual(F)

"""
    projection_residual(form::AbstractProblemForm, F::AbstractMatrix) -> AbstractMatrix

`F - Π_C(F)` for every column of constraint values `F` (all rows), where `C` is
the product of the constraint sets: intervals `[ℓᵢ, uᵢ]` for box-like rows and
second-order cones for the blocks of [`cone_blocks`](@ref). Equality rows give
`h`, inequality rows give `ReLU(g)` (as a signed distance), and cone blocks give
the cone residual. FSNet's violation is `φ = ‖F - Π_C(F)‖²`, and its gradient
with respect to `F` is `2(F - Π_C(F))`.
"""
function projection_residual(form::AbstractProblemForm, F::AbstractMatrix)
    lower, upper = row_bounds(form)
    R = _projection_residual(F, lower, upper)
    blocks = cone_blocks(form)
    isempty(blocks) && return R
    m = size(F, 1)
    segs = _nograd(() -> _segments(m, blocks))
    parts = map(segs) do (rows, kind)
        kind === :box ? R[rows, :] : _cone_residual(kind, F[rows, :])
    end
    return reduce(vcat, parts)
end

# Distance of each cone block to its cone (n_cones × B).
function _cone_block_distances(form::AbstractProblemForm, F::AbstractMatrix)
    blocks = cone_blocks(form)
    isempty(blocks) && return similar(F, 0, size(F, 2))
    return reduce(
        vcat, [sqrt.(_colsumsq(_cone_residual(k, F[r, :])))' for (r, k) in blocks]
    )
end
