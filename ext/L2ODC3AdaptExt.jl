module L2ODC3AdaptExt

# Move the dense affine DC3/FSNet data to an accelerator (e.g.
# `adapt(CuArray, d)` or `adapt(CuArray{Float32}, d)`). Scalars are converted to
# the element type of the adapted arrays so no mixed-precision promotion
# happens on the device. Index vectors move too (row gathers stay on device);
# masks and the independent/dependent index lists stay on the host.

using Adapt: Adapt
using L2ODC3: AffineForm, LinearCompletion, DC3Decomposition, FSNetDecomposition

function Adapt.adapt_structure(to, f::AffineForm)
    A = Adapt.adapt(to, f.A)
    T = eltype(A)
    return AffineForm(
        T(f.sense),
        A,
        Adapt.adapt(to, f.B),
        Adapt.adapt(to, f.b),
        Adapt.adapt(to, f.lower),
        Adapt.adapt(to, f.upper),
        Adapt.adapt(to, f.Q),
        Adapt.adapt(to, f.c),
        Adapt.adapt(to, f.H),
        Adapt.adapt(to, f.cp),
        Adapt.adapt(to, f.Qpp),
        T(f.c0),
        f.eq,
        f.bnd,
        _adapt_index(to, f.eqidx),
        _adapt_index(to, f.inidx),
    )
end

# `adapt(CuArray{Float32}, ints)` would convert integers to floats; move index
# vectors with the array constructor only.
_adapt_index(to, idx) = Adapt.adapt(_storage(to), idx)
_storage(to::Type{<:AbstractArray}) = Base.typename(to).wrapper
_storage(to) = to

function Adapt.adapt_structure(to, c::LinearCompletion)
    return LinearCompletion(
        c.independent,
        c.dependent,
        Adapt.adapt(to, c.Kz),
        Adapt.adapt(to, c.Kp),
        Adapt.adapt(to, c.k0),
    )
end

function Adapt.adapt_structure(to, d::DC3Decomposition)
    form = Adapt.adapt(to, d.form)
    return DC3Decomposition(
        form, Adapt.adapt(to, d.completion), eltype(form.A)(d.step_size)
    )
end

function Adapt.adapt_structure(to, d::FSNetDecomposition)
    form = Adapt.adapt(to, d.form)
    return FSNetDecomposition(form, eltype(form.A)(d.step_size))
end

end # module L2ODC3AdaptExt
