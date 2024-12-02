using TeneT: simple_eig
using VectorInterface
using BenchmarkTools
using CUDA
using LinearAlgebra
using Test
using ProfileView
using Random
using Zygote
using KrylovKit
using Krylov
using OMEinsum

CUDA.allowscalar(false)
@testset "dot $atype" for atype in [Array, CuArray]
    Random.seed!(100)
    N = 10^2
    A = atype(rand(ComplexF64, N,N))
    B = [A]
    C = [B]
    D = [C]
    # @btime CUDA.@sync inner($A, $A)
    # @btime CUDA.@sync inner($B, $B)
    # @btime CUDA.@sync inner($C, $C)
    # @btime CUDA.@sync inner($D, $D)
    function foo(x)
        for _ in 1:100
            CUDA.@sync inner(x, x)
        end
        
    end
    ProfileView.@profview foo(D)
end

@testset "dot $atype" for atype in [Array, CuArray]
    Random.seed!(100)
    N = 10^2
    A = atype(rand(ComplexF64, N,N))
    B = [A]
    C = [B]
    D = [C]
    @btime CUDA.@sync inner($A, $A)
    @btime CUDA.@sync inner($B, $B)
    @btime CUDA.@sync inner($C, $C)
    @btime CUDA.@sync inner($D, $D)
end

@testset "eigsolve $atype" for atype in [Array, CuArray]
    Random.seed!(100)
    N = 10^3
    A = [atype(rand(ComplexF64, N, N)) for i in 1:4]
    v0 = [atype(rand(ComplexF64, N)) for i in 1:4]
    linearmap(v) = A .* v
    @btime CUDA.@sync inner($v0, $v0)
    @btime CUDA.@sync $linearmap($v0)
    @btime CUDA.@sync λs, vs = eigsolve(v -> $linearmap(v), $v0, 1, :LM)
end
    # function foo()
    #     for _ in 1:100
    #         λs, vs = eigsolve(v -> linearmap(v), v0, 1, :LM)
    #     end
    # end
    # ProfileView.@profview foo()
@testset "eigsolve $atype N=$N" for atype in [Array, CuArray], N in [10^3, 5*10^3]
    Random.seed!(100)
    A = atype(rand(ComplexF64, N, N))
    v0 = atype(rand(ComplexF64, N))
    linearmap(v) = A * v
    @btime CUDA.@sync $A * $v0
    @btime CUDA.@sync λs, vs = eigsolve(v -> $linearmap(v), $v0, 1, :LM)
end

@testset "eigsolve $atype" for atype in [Array]
    Random.seed!(100)
    N = 10^3
    A = [atype(rand(ComplexF64, N, N)) for i in 1:1]
    v0 = [atype(rand(ComplexF64, N, N)) for i in 1:1]
    linearmap(v) = A .* v
    # @btime CUDA.@sync $A .* $v0
    # @btime CUDA.@sync λs, vs = eigsolve(v -> $linearmap(v), $v0, 1, :LM)
    # @btime CUDA.@sync $simple_eig($linearmap, $v0)
    λ, v = simple_eig(linearmap, v0)
    @show λ norm(λ*v - linearmap(v))

    # ProfileView.@profview λs, vs = eigsolve(v -> A*v, v0, 1, :LM)
end

@testset "linsolve $atype" for atype in [Array, CuArray]
    Random.seed!(100)
    N = 10^2
    A = atype(rand(ComplexF64, N, N))
    b = atype(rand(ComplexF64, N))
    # @btime CUDA.@sync $A \ $b
    # @btime CUDA.@sync linsolve(X -> $A * X, $b; maxiter = 100)
    # @btime CUDA.@sync x, stats = bilq($A, $b)
    # ProfileView.@profview linsolve(X -> A * X, b; maxiter = 100)
    # x, stats = bilq(X -> A * X, b)
end