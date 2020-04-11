module TestMetrics
using Test
using LinearAlgebra
using Metrics
using Metrics: EuclideanInner, EuclideanNorm, EnergeticNorm, PNorm
using LinearAlgebraExtensions: LowRank

@testset "Metrics" begin

    n = 3
    x = randn(n)
    y = randn(n)
    @test EuclideanInner()(x, y) == dot(x, y)

    using ForwardDiff: derivative, gradient
    d, r = 2, 1 # data input dimension and rank of energetic norm
    U = randn(d, r)
    enorm = EnergeticNorm(LowRank(U))
    x = randn(d)
    # 1. differentiate input
    @test gradient(enorm, x) isa Vector
    @test gradient(EuclideanNorm(), zeros(d)) ≈ zeros(d)
    @test gradient(PNorm{Float64}(1+rand()), zeros(d)) ≈ zeros(d)
    @test gradient(enorm, zeros(d)) ≈ zeros(d)

    # 2. differentiate matrix
    function f(θ)
        U = reshape(θ, d, r)
        EnergeticNorm(LowRank(U))(zeros(d))
    end
    θ = U[:]
    @test gradient(f, θ) ≈ zero(θ)
end

end # TestMetrics
