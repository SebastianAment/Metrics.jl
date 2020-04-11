##################### Efficient Implementations of Distances  ##########################
module Metrics
using LinearAlgebraExtensions: difference, AbstractMatOrFac
using LinearAlgebra: dot
using ForwardDiff
const FD = ForwardDiff

# TODO: could define everything below for functions!

abstract type Metric{T} end
abstract type Norm{T} end #<: Metric{T} end
abstract type InnerProduct{T} end #<: Norm{T} end

_atypes = [:InnerProduct, :Norm, :Metric]

InnerProduct(i::InnerProduct) = i
Norm(n::Norm) = n
Metric(m::Metric) = m

# fallbacks, if we separate modules, could have only those which make sense
(N::Norm)(x) = sqrt(InnerProduct(N)(x, x))
(M::Metric)(x, y) = Norm(M)(difference(x, y))

################################ Euclidean #####################################
_euc = [:EuclideanInner, :EuclideanNorm, :EuclideanMetric]

for (sym, atype) in zip(_euc, _atypes)
    @eval struct $sym{T} <: $atype{T} end
    @eval $sym() = $sym{Float64}()
end

const Euclidean{T} = Union{EuclideanInner{T}, EuclideanNorm{T}, EuclideanMetric{T}}

for (sym, atype) in zip(_euc, _atypes)
    @eval $atype(::Euclidean) = $sym()
end

(::EuclideanInner)(x, y) = dot(x, y)

# TODO: this special case might not be necessary for performance
(::EuclideanNorm)(x) = √sum(abs2, x)

################################### p-norm #####################################
_p = [:PNorm, :PMetric]
for (sym, atype) in zip(_p, _atypes[2:3])
    eval(quote
        struct $sym{T} <: $atype{T}
            p::T
            $sym{T}(p::T) where {T} = 1 ≤ p ? new(p) : error("p is smaller than 1")
        end
    end)
end

Norm(n::PMetric) = PNorm(n.p)
Metric(n::PNorm) = PMetric(n.p)

(l::PNorm)(x) = sum(x->x^l.p, x)^(1/l.p)

######################## Energetic Inner Product ###############################
_energetic = [:EnergeticInner, :EnergeticNorm, :EnergeticMetric]

using LinearAlgebra: Factorization

# Energetic, Hermitian Norm, because we have a Hermitian matrix
# check for postive semi-definiteness of A (positive diagonal at least)
for (sym, atype) in zip(_energetic, _atypes)
    eval(quote
        struct $sym{T, M<:AbstractMatOrFac{T}} <: $atype{T}
            A::M
        end
    end)
end

const Energetic{T} = Union{EnergeticInner{T}, EnergeticNorm{T}, EnergeticMetric{T}}

for (sym, atype) in zip(_energetic, _atypes)
    @eval $atype(E::Energetic) = $sym(E.A)
end

(E::EnergeticInner)(x, y) = dot(x, E.A, y) # energetic inner product

########################### for AD , avoids NaN 0 ##############################
# i.e. makes forwardiff for norms behave like sum(abs, 0) at 0
@inline function _fd_euclidean_norm(x)
    all(==(0), x) ? zero(eltype(x)) : √sum(abs2, x)
end
@inline function _fd_p_norm(p, x)
    all(==(0), x) ? zero(eltype(x)) : sum(x->x^p, x)^(1/p)
end
@inline function _fd_energetic_norm(A, x)
    all(==(0), x) ? zero(eltype(x)) : √dot(x, A, x)
end

using ForwardDiff: Dual
const DualContainer{T<:Dual} = Union{T, AbstractArray{T}, NTuple{N, T} where N}

(N::EuclideanNorm)(x::DualContainer) = _fd_euclidean_norm(x)

(N::PNorm)(x::DualContainer) = _fd_p_norm(N.p, x)
(N::PNorm{<:Dual})(x) = _fd_p_norm(N.p, x)

(N::EnergeticNorm)(x::DualContainer) = _fd_energetic_norm(N.A, x)
(N::EnergeticNorm{<:Dual})(x) = _fd_energetic_norm(N.A, x)

end # Metrics
