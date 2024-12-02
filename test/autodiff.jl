using TeneT
using TeneT: _arraytype
using TeneT: qrpos,lqpos,left_canonical,leftenv,right_canonical,rightenv,ACenv,Cenv,LRtoC,ALCtoAC,ACCtoALAR,env_norm
using TeneT: vumps_step
using ChainRulesCore
using CUDA
using LinearAlgebra
using OMEinsum
using Random
using Test
using Zygote
CUDA.allowscalar(false)

begin "test utils"
    function num_grad(f, K; δ::Real=1e-5)
        if eltype(K) == ComplexF64
            (f(K + δ / 2) - f(K - δ / 2)) / δ + 
                (f(K + δ / 2 * 1.0im) - f(K - δ / 2 * 1.0im)) / δ * 1.0im
        else
            (f(K + δ / 2) - f(K - δ / 2)) / δ
        end
    end
    
    function num_grad(f, a::AbstractArray; δ::Real=1e-5)
        b = Array(copy(a))
        df = map(CartesianIndices(b)) do i
            foo = x -> (ac = copy(b); ac[i] = x; f(_arraytype(a)(ac)))
            num_grad(foo, b[i], δ=δ)
        end
        return _arraytype(a)(df)
    end
end

@testset "zygote mutable arrays with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64]
    Random.seed!(100)
    function foo(F) 
        buf = Zygote.Buffer(F) # https://fluxml.ai/Zygote.jl/latest/utils/#Zygote.Buffer
        @inbounds @views for j in 1:2, i in 1:2 
            buf[:,:,:,i,j] = F[:,:,:,i,j]./norm(F[:,:,:,i,j]) 
        end
        return norm(copy(buf))
    end
    F = atype(rand(dtype, 3,2,3,2,2))
    @test Zygote.gradient(foo, F)[1] ≈ num_grad(foo, F) atol = 1e-8
end

@testset "loop_einsum mistake with $atype{$dtype}" for atype in [CuArray], dtype in [ComplexF64]
    Random.seed!(100)
    D = 5
    A = atype(rand(dtype, D,D,D))
    B = atype(rand(dtype, D,D))
    function foo(x)
        C = A * x
        D = B * x
        E = Array(ein"abc,abc -> "(C,C))[]
        F = Array(ein"ab,ab -> "(D,D))[]
        return norm(E/F)
        # E = ein"abc,abc -> "(C,C)[]
        # F = ein"ab,ab -> "(D,D)[]
        # return norm(E/F) mistake for GPU
    end 
    @test Zygote.gradient(foo, 1)[1] ≈ num_grad(foo, 1) atol = 1e-8
end

@testset "QR factorization with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64]
    Random.seed!(100)
    M = atype(rand(dtype, 3, 3))
    function foo(M)
        Q, R = qrpos(M)
        return norm(Q) + norm(R)
    end
    @test Zygote.gradient(foo, M)[1] ≈ num_grad(foo, M) atol = 1e-8
end

@testset "LQ factorization with $atype{$dtype}" for atype in [Array], dtype in [ComplexF64]
    Random.seed!(100)
    M = atype(rand(dtype, 3, 3))
    function foo(M)
        L, Q = lqpos(M)
        return  norm(Q) + norm(L)
    end
    @test Zygote.gradient(foo, M)[1] ≈ num_grad(foo, M) atol = 1e-8
end

@testset "$(Ni)x$(Nj) leftenv and rightenv with $atype" for atype in [Array], ifobs in [false], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 3, 2
    A = [atype(rand(ComplexF64, D, d, D         )) for _ in 1:Ni, _ in 1:Nj]
    S = [atype(rand(ComplexF64, D, d, D, D, d, D)) for _ in 1:Ni, _ in 1:Nj]
    M = [atype(rand(ComplexF64, d, d, d, d      )) for _ in 1:Ni, _ in 1:Nj]

       ALu, =  left_canonical(A) 
       ALd, =  left_canonical(A)
    _, ARu, = right_canonical(A)
    _, ARd, = right_canonical(A)

    function foo1(M)
        _,FL = leftenv(ALu, conj(ALd), M; ifobs)
        s = 0.0
        for j in 1:Nj, i in 1:Ni
            A  = Array(ein"(abc,abcdef),def -> "(FL[i,j], S[i,j], FL[i,j]))[]
            B  = Array(ein"abc,abc -> "(FL[i,j], FL[i,j]))[]
            s += norm(A/B)
        end
        return s
    end 
    @test Zygote.gradient(foo1, M)[1] ≈ num_grad(foo1, M) atol = 1e-7

    function foo2(M)
        _,FR = rightenv(ARu, conj(ARd), M; ifobs)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A  = Array(ein"(abc,abcdef),def -> "(FR[i,j], S[i,j], FR[i,j]))[]
            B  = Array(ein"abc,abc -> "(FR[i,j], FR[i,j]))[]
            s += norm(A/B)
        end
        return s
    end 
    @test Zygote.gradient(foo2, M)[1] ≈ num_grad(foo2, M) atol = 1e-7
end

@testset "$(Ni)x$(Nj) ACenv and Cenv with $atype" for atype in [Array], dtype in [ComplexF64], Ni = [2], Nj = [2]
    Random.seed!(100)
    D, d = 3, 2
     A = [atype(rand(ComplexF64, D, d, D,        )) for _ in 1:Ni, _ in 1:Nj] 
    S1 = [atype(rand(ComplexF64, D, d, D, D, d, D)) for _ in 1:Ni, _ in 1:Nj]
    S2 = [atype(rand(ComplexF64, D, D, D, D,     )) for _ in 1:Ni, _ in 1:Nj]
     M = [atype(rand(ComplexF64, d, d, d, d,     )) for _ in 1:Ni, _ in 1:Nj]

    AL, L, _ =  left_canonical(A) 
    R, AR, _ = right_canonical(A)
    _, FL    =  leftenv(AL, conj(AL), M)
    _, FR    = rightenv(AR, conj(AR), M)

     C =   LRtoC( L, R)
    AC = ALCtoAC(AL, C)
    function foo1(M)
        _, AC = ACenv(AC, FL, M, FR)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = Array(ein"(abc,abcdef),def -> "(AC[i,j], S1[i,j], AC[i,j]))[]
            B = Array(ein"abc,abc -> "(AC[i,j], AC[i,j]))[]
            s += norm(A/B)
        end
        return s
    end
    @test Zygote.gradient(foo1, M)[1] ≈ num_grad(foo1, M) atol = 1e-7

    function foo2(M)
        _, FL = leftenv(AL, conj(AL), M)
        _, FR = rightenv(AR, conj(AR), M)
        _, C = Cenv(C, FL, FR)
        s = 0
        for j in 1:Nj, i in 1:Ni
            A = Array(ein"(ab,abcd),cd -> "(C[i,j], S2[i,j], C[i,j]))[]
            B = Array(ein"ab,ab -> "(C[i,j], C[i,j]))[]
            s += norm(A/B)
        end
        return s
    end
    @test Zygote.gradient(foo2, M)[1] ≈ num_grad(foo2, M) atol = 1e-7
end

@testset "$(Ni)x$(Nj) ACCtoALAR with $atype" for atype in [Array], Ni = [1], Nj = [1]
    Random.seed!(100)
    D, d = 3, 2
    A =  [atype(rand(ComplexF64, D, d, D,        )) for _ in 1:Ni, _ in 1:Nj] 
    S1 = [atype(rand(ComplexF64, D, d, D, D, d, D)) for _ in 1:Ni, _ in 1:Nj]
    S2 = [atype(rand(ComplexF64, D, D, D, D,     )) for _ in 1:Ni, _ in 1:Nj]
     M = [atype(rand(ComplexF64, d, d, d, d,     )) for _ in 1:Ni, _ in 1:Nj]

    AL, L, _ =  left_canonical(A) 
    R, AR, _ = right_canonical(A)
    _, FL    =  leftenv(AL, conj(AL), M)
    _, FR    = rightenv(AR, conj(AR), M)

     Co =   LRtoC( L, R)
    ACo = ALCtoAC(AL, Co)
    _, Co = Cenv(Co, FL, FR)
    function foo1(M)
        _, AC = ACenv(ACo, FL, M, FR)
        AL, AR = ACCtoALAR(AC, Co) 
        s = 0
        for j in 1:Nj, i in 1:Ni
            A  = Array(ein"(abc,abcdef),def -> "(AL[i,j], S1[i,j], AL[i,j]))[]
            B  = Array(ein"abc,abc -> "(AL[i,j], AL[i,j]))[]
            s += norm(A/B)
            A  = Array(ein"(abc,abcdef),def -> "(AR[i,j], S1[i,j], AR[i,j]))[]
            B  = Array(ein"abc,abc -> "(AR[i,j], AR[i,j]))[]
            s += norm(A/B)
            A  = Array(ein"(abc,abcdef),def -> "(AC[i,j], S1[i,j], AC[i,j]))[]
            B  = Array(ein"abc,abc -> "(AC[i,j], AC[i,j]))[]
            s += norm(A/B)
        end
        return s
    end
    @test Zygote.gradient(foo1, M)[1] ≈ num_grad(foo1, M) atol = 1e-4
end

@testset "$(Ni)x$(Nj) fix_gauge_vumps_step backward with $atype" for Ni in 1:1, Nj in 1:1, atype = [Array]
    Random.seed!(100)

    alg = VUMPS(maxiter=200, miniter=100, verbosity=3, ifupdown=false)
    χ = 3
    β = 0.2
    model = Ising(Ni, Nj, β)
    M = atype.(model_tensor(model, Val(:bulk)))
    rt = VUMPSRuntime(M, χ, alg)
    rt = leading_boundary(rt, M, alg)

    function energy(β)
        model = Ising(Ni, Nj, β)
        M = atype.(model_tensor(model, Val(:bulk)))
        rt′, _ = fix_gauge_vumps_step(rt, M, alg)
        env = VUMPSEnv(rt′, M)
        return real(observable(env, model, Val(:energy)))
    end
    @test Zygote.gradient(energy, β)[1] ≈ num_grad(energy, β)
end

@testset "$(Ni)x$(Nj) ising backward with $atype" for Ni in 1:1, Nj in 1:1, atype = [Array]
    Random.seed!(100)

    alg = VUMPS(maxiter=100, miniter=50, verbosity=3, ifupdown=false)
    χ = 10
    β = 0.5
    model = Ising(Ni, Nj, β)
    M = atype.(model_tensor(model, Val(:bulk)))
    rt = VUMPSRuntime(M, χ, alg)

    function energy(β)
        model = Ising(Ni, Nj, β)
        M = atype.(model_tensor(model, Val(:bulk)))
        rt′ = leading_boundary(rt, M, alg)
        env = VUMPSEnv(rt′, M)
        return real(observable(env, model, Val(:energy)))
    end
    # @test Zygote.gradient(energy, 0.5)[1] ≈ num_grad(energy, 0.5)
    @show Zygote.gradient(energy, 0.5)[1]
    # @show norm(Zygote.gradient(energy, 0.5)[1] - num_grad(energy, 0.5))
end

using KrylovKit
@testset "reallinsolve" begin
    Random.seed!(100)
    A = [rand(ComplexF64, 3,3) for _ in 1:2]
    b = [rand(ComplexF64, 3) for _ in 1:2]
    x, info = reallinsolve(x->A .* x, b, b, GMRES())
    @show x
    @test norm(A .* x - b) < 1e-12

    x, info = linsolve(x->A .* x, b, b, GMRES())
    @show x
    @test norm(A .* x - b) < 1e-12
end