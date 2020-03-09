using Test
using LinearAlgebra
using Metrics: EuclideanInner
n = 3
x = randn(n)
y = randn(n)
@test EuclideanInner()(x, y) == dot(x, y)
