begin 
    using TeneT
    using U1ArrayKit
    using TeneT:qrpos,lqpos,left_canonical,right_canonical,leftenv,FLmap,rightenv,FRmap,ACenv,ACmap,Cenv,Cmap,LRtoC,ALCtoAC,ACCtoALAR,error, env_norm
    using TeneT:_to_front, _to_tail, permute_fronttail
    using TeneT: qrpos,lqpos,left_canonical,leftenv,right_canonical,rightenv,ACenv,Cenv,LRtoC,ALCtoAC,ACCtoALAR,env_norm, fix_gauge_vumps_step
    using ChainRulesCore
    using CUDA
    using LinearAlgebra
    using OMEinsum
    using Random
    using Test
    using Zygote
    CUDA.allowscalar(false)
end

begin "test utils"
    test_type = [Array]
    χ, D, d = 4, 2, 2
    # test_As = [rand(ComplexF64, χ, D, χ), rand(ComplexF64, χ, D, D, χ)];
    # test_Ms = [rand(ComplexF64, D, D, D, D), rand(ComplexF64, D, D, D, D, d)];
    test_As = [randU1double(Array, χ,D^2,χ)];
    m = randU1double(Array, D^2,D^2,D^2,D^2)
    m += permutedims(m, [1,4,3,2])
    m += permutedims(m, [3,2,1,4])
    normalize!(m)
    test_Ms = [m];
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

@testset "TeneT.jl" begin
    @testset "environment" begin
        println("environment tests running...")
        include("environment.jl")
    end

    @testset "vumpsruntime.jl" begin
        println("vumpsruntime tests running...")
        include("vumpsruntime.jl")
    end

    @testset "autodiff.jl" begin
        println("autodiff tests running...")
        include("autodiff.jl")
    end
end
