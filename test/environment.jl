@testset "qr with $atype{$dtype}" for atype in test_type, dtype in [Float64, ComplexF64]
    Random.seed!(100)
    A = atype(rand(dtype, 4,4))
    Q, R = qrpos(A)
    @test Array(Q*R) ≈ Array(A)
    @test all(real.(diag(R)) .> 0)
    @test all(imag.(diag(R)) .≈ 0)
end

@testset "lq with $atype{$dtype}" for atype in test_type, dtype in [Float64, ComplexF64]
    Random.seed!(100)
    A = atype(rand(dtype, 4,4))
    L, Q = lqpos(A)
    @test Array(L*Q) ≈ Array(A)
    @test all(real.(diag(L)) .> 0)
    @test all(imag.(diag(L)) .≈ 0)
end

@testset "left_canonical and right_canonical with $atype $Ni x $Nj" for atype in test_type, a in test_As, Ni in 1:3, Nj in 1:3
    Random.seed!(100)
    
    A = [atype(a) for i in 1:Ni, j in 1:Nj]
    AL,  L, λ =  left_canonical(A)
     R, AR, λ = right_canonical(A)
     
    for j = 1:Nj,i = 1:Ni
        @test asComplexArray(asArray(DoublePEPSZ2(Int(sqrt(χ))), Array(_to_tail(AL[i,j])' * _to_tail(AL[i,j])))) ≈ I(χ)

        LA = _to_tail(L[i,j] * _to_front(A[i,j]))
        ALL = _to_tail(AL[i,j]) * L[i,j] * λ[i,j]
        @test (Array(ALL) ≈ Array(LA))

        @test asComplexArray(asArray(DoublePEPSZ2(Int(sqrt(χ))), Array(_to_front(AR[i,j]) * _to_front(AR[i,j])'))) ≈ I(χ)

        AxR = _to_front(_to_tail(A[i,j]) * R[i,j])
        RAR = R[i,j] * _to_front(AR[i,j]) * λ[i,j]
        @test (Array(RAR) ≈ Array(AxR))
    end
end

@testset "leftenv and rightenv with $atype $Ni x $Nj" for atype in test_type, (a, m) in zip(test_As, test_Ms), ifobs in [false], Ni in 1:1, Nj in 1:1
    Random.seed!(100)
    A = [atype(a) for i in 1:Ni, j in 1:Nj]
    # M = [atype(m) for i in 1:Ni, j in 1:Nj]
    ipeps = rand(ComplexF64, D,D,D,D,d)
    # ipeps += permutedims(conj(ipeps), (3,2,1,4,5))
    # ipeps += permutedims(conj(ipeps), (1,4,3,2,5))
    normalize!(ipeps)
    M = reshape(ein"abcdi,efghi->aebfcgdh"(ipeps, conj(ipeps)), D^2,D^2,D^2,D^2)
    M = convert_bilayer_Z2(DoubleArray(M))
    ST = SymmetricType(symmetry=:U1, stype=DoublePEPSZ2(D), atype=Array, dtype=Float64)
    M = asSymmetryArray(M, ST; dir=[-1,1,1,-1])
    # @show M.imag
    M = [M for i in 1:Ni, j in 1:Nj]

    alg = VUMPS(ifsimple_eig=false)
    AL,    =  left_canonical(A)
    M = [asArray([DoublePEPSZ2(s) for s in (D,D,D,D)], asComplexArray(M[i,j])) for i in 1:Ni, j in 1:Nj]
    AL = [asArray([DoublePEPSZ2(s) for s in (Int(sqrt(χ)),D,Int(sqrt(χ)))], asComplexArray(AL[i,j])) for i in 1:Ni, j in 1:Nj]
    λL,FL  =  leftenv(AL, conj(AL), M; alg, ifobs)
    # _, AR, = right_canonical(A)
    # λR,FR  = rightenv(AR, conj(AR), M; alg, ifobs)

    for i in 1:Ni
        ir = ifobs ? Ni+1-i : mod1(i+1, Ni)
        for j in 1:Nj
            @show λL[i] norm(λL[i])
            @show norm(λL[i] * FL[i,j] - FLmap(j, FL[i,j], AL[i,:], conj.(AL)[ir,:], M[i,:]))
            # @test λL[i] * FL[i,j] ≈ FLmap(j, FL[i,j], AL[i,:], conj.(AL)[ir,:], M[i,:]) rtol = 1e-12
            # @test λR[i] * FR[i,j] ≈ FRmap(j, FR[i,j], AR[i,:], conj.(AR)[ir,:], M[i,:]) rtol = 1e-12
        end
    end
end

@testset "ACenv and Cenv with $atype $Ni x $Nj" for atype in test_type, (a, m) in zip(test_As, test_Ms), ifobs in [false, true], Ni in 1:3, Nj in 1:3
    Random.seed!(100)
    A = [atype(a) for i in 1:Ni, j in 1:Nj]
    M = [atype(m) for i in 1:Ni, j in 1:Nj]

    AL,  L, _ =  left_canonical(A)
     R, AR, _ = right_canonical(A)
    λL, FL    =  leftenv(AL, conj(AL), M)
    λR, FR    = rightenv(AR, conj(AR), M)

     C =  LRtoC(  L, R)
    AC = ALCtoAC(AL, C)

    λAC, AC = ACenv(AC, FL, M, FR)
     λC,  C =  Cenv( C, FL,    FR)

    for j in 1:Nj
        jr = mod1(j + 1, Nj)
        for i in 1:Ni
            ir = mod1(i + 1, Ni)
            @test λAC[j] * AC[i,j] ≈ ACmap(i, AC[i,j], FL[:,j],  FR[:,j], M[:,j]) rtol = 1e-12
            @test  λC[j] *  C[i,j] ≈  Cmap(i,  C[i,j], FL[:,jr], FR[:,j]) rtol = 1e-10
        end
    end
end

@testset "bcvumps unit test with $atype $Ni x $Nj" for atype in test_type, (a, m) in zip(test_As, test_Ms), ifobs in [false, true], Ni in 1:3, Nj in 1:3
    Random.seed!(100)
    A = [atype(a) for i in 1:Ni, j in 1:Nj]
    M = [atype(m) for i in 1:Ni, j in 1:Nj]

    AL,  L, _ =  left_canonical(A)
     R, AR, _ = right_canonical(A)
    λL, FL    =  leftenv(AL, conj(AL), M)
    λR, FR    = rightenv(AR, conj(AR), M)

    C = LRtoC(L,R)
    AC = ALCtoAC(AL,C)
    
    λAC, AC = ACenv(AC, FL, M, FR)
     λC,  C =  Cenv( C, FL,    FR)
    AL, AR, errL, errR = ACCtoALAR(AC, C)
    @test errL + errR !== nothing
end