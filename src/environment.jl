"""
    VUMPSEnv{T<:Number, S<:IndexSpace,
             OT<:AbstractArray{S, 2, 2},
             ET<:AbstractArray{S, 2, 1},
             CT<:AbstractArray{S, 1, 1}}

A struct that contains the environment of the VUMPS algorithm for calculate observables.
    
For a `Ni` x `Nj` unitcell, each is a Matrix, containing

- `AC`: The mixed canonical environment tensor.
- `AR`: The right canonical environment tensor.
- `Lu`: The left upper environment tensor.
- `Ru`: The right upper environment tensor.
- `Lo`: The left mixed environment tensor.
- `Ro`: The right mixed environment tensor.
"""
struct VUMPSEnv{T<:Number,
                ET<:AbstractArray}
    ACu::Matrix{ET}
    ARu::Matrix{ET}
    ACd::Matrix{ET}
    ARd::Matrix{ET}
    FLu::Matrix{ET}
    FRu::Matrix{ET}
    FLo::Matrix{ET}
    FRo::Matrix{ET}
    function VUMPSEnv(ACu::Matrix{ET},
                      ARu::Matrix{ET},
                      ACd::Matrix{ET},
                      ARd::Matrix{ET},
                      FLu::Matrix{ET},
                      FRu::Matrix{ET},
                      FLo::Matrix{ET},
                      FRo::Matrix{ET}) where {ET}
        T = eltype(ACu[1])
        new{T, ET}(ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo)
    end
end

"""
    VUMPSRuntime{T<:Number, S<:IndexSpace,
                 OT<:AbstractArray{S, 2, 2},
                 ET<:AbstractArray{S, 2, 1},
                 CT<:AbstractArray{S, 1, 1}}

A struct that contains the environment of the VUMPS algorithm for runtime calculations.
    
For a `Ni` x `Nj` unitcell, each is a Matrix, containing

- `AL`: The left canonical environment tensor.
- `AR`: The right canonical environment tensor.
- `C`: The canonical environment tensor.
- `L`: The left environment tensor.
- `R`: The right environment tensor.
"""
struct VUMPSRuntime             
    AL
    AR
    C
    FL
    FR
end

# In-place update of environment
function update!(env::VUMPSRuntime, env´::VUMPSRuntime) 
    env.AL .= env´.AL
    env.AR .= env´.AR
    env.C .= env´.C
    env.FL .= env´.FL
    env.FR .= env´.FR
    return env
end

function update!(env::Tuple{VUMPSRuntime, VUMPSRuntime}, env´::Tuple{VUMPSRuntime, VUMPSRuntime}) 
    update!(env[1], env´[1]) 
    update!(env[2], env´[2])
    return env
end

function update!(env::VUMPSRuntime, env´::Tuple{VUMPSRuntime, VUMPSRuntime}) 
    update!(env, env´[1])
    return env
end

Array(rt::VUMPSRuntime) = VUMPSRuntime(Array.(rt.AL), Array.(rt.AR), Array.(rt.C), Array.(rt.FL), Array.(rt.FR))
Array(rt::Tuple{VUMPSRuntime, VUMPSRuntime}) = Array.(rt)
CuArray(rt::VUMPSRuntime) = VUMPSRuntime(CuArray.(rt.AL), CuArray.(rt.AR), CuArray.(rt.C), CuArray.(rt.FL), CuArray.(rt.FR))
CuArray(rt::Tuple{VUMPSRuntime, VUMPSRuntime}) = CuArray.(rt)

"""
tensor order graph: from left to right, top to bottom.
```
a ────┬──── c    a──────┬──────c     a─────b
│     b     │    │      │      │     │     │
├─ d ─┼─ e ─┤    │      b      │     ├──c──┤           
│     g     │    │      │      │     │     │
f ────┴──── h    d──────┴──────e     d─────e
```
"""
function env_norm(F::Matrix)
    return [F/norm(F) for F in F]
end

"""
    λs[1], Fs[1] = selectpos(λs, Fs)

Select the max positive one of λs and corresponding Fs.
"""
function selectpos(λs, Fs, N)
    if length(λs) > 1 && norm(abs(λs[1]) - abs(λs[2])) < 1e-12
        # @show "selectpos: λs are degeneracy"
        N = min(N, length(λs))
        p = argmax(real(λs[1:N]))  
        # @show λs p abs.(λs)
        return λs[1:N][p], Fs[1:N][p]
    else
        return λs[1], Fs[1]
    end
end

function initial_A(M::leg4, χ::Int; kwargs...)
    Ni, Nj = size(M)
    atype = _arraytype(M[1])
    return [(D = size(M[i,j], 4); atype(rand(ComplexF64, χ,D,χ))) for i = 1:Ni, j = 1:Nj]
end

function initial_A(M::leg5, χ::Int; kwargs...)
    Ni, Nj = size(M)
    atype = _arraytype(M[1])
    return [(D = size(M[i,j], 4); atype(rand(ComplexF64, χ,D,D,χ))) for i = 1:Ni, j = 1:Nj]
end

function initial_A(M::leg8, χ::Int; kwargs...)
    Ni, Nj = size(M)
    atype = _arraytype(M[1])
    return [(D = size(M[i,j], 7); atype(rand(ComplexF64, χ,D,D,χ))) for i = 1:Ni, j = 1:Nj]
end

function initial_A(M::Matrix{<:U1Array}, χ::Int; alg)
    Ni, Nj = size(M)
    atype = _arraytype(M[1])
    A = Array{atype{ComplexF64, 4}, 2}(undef, Ni, Nj)
    qnD, qnχ, dimsD, dimsχ = alg.U1info
    indqn = [qnχ, qnD, qnD, qnχ]
    indims = [dimsχ, dimsD, dimsD, dimsχ]
    D = sum(dimsD)
    for j in 1:Nj, i in 1:Ni
        A[i,j] = randinitial(M[i,j], χ, D, D, χ; 
                             dir = [-1, -1, 1, 1], indqn = indqn, indims = indims
        )
    end
    return A
end

function cellones(A; kwargs...)
    Ni, Nj = size(A)
    χ = size(A[1], 1)
    atype = _arraytype(A[1])
    return [atype{ComplexF64}(I, χ, χ) for _ = 1:Ni, _ = 1:Nj]
end

function cellones(A::Matrix{<:U1Array}; alg)
    Ni, Nj = size(A)
    χ = size(A[1,1],1)
    atype = _arraytype(A[1,1])
    C = Array{atype, 2}(undef, Ni, Nj)
    _, qnχ, _, dimsχ = alg.U1info
    dir = getdir(A[1,1])[[1,3]]
    for j in 1:Nj, i in 1:Ni
        C[i,j] = Iinitial(A[i,j], χ; 
                          dir = dir, indqn = [qnχ, qnχ], indims = [dimsχ, dimsχ]  
        )
    end
    return C
end

ρmap(ρ, Au::leg3, Ad::leg3) = ein"(dc,csb),dsa -> ab"(ρ,Au,Ad)
ρmap(ρ, Au::leg4, Ad::leg4) = ein"(dc,cstb),dsta -> ab"(ρ,Au,Ad)

function ρmap(ρ, Ai, J::Int)
    Nj = size(Ai,1)
    for j = 1:Nj
        jr = mod1(J+j-1, Nj)
        ρ = ρmap(ρ,Ai[jr],conj(Ai[jr]))
    end
    return ρ
end

"""
    getL!(A,L; kwargs...)

````
┌─ Aᵢⱼ ─ Aᵢⱼ₊₁─     ┌─      L ─
ρᵢⱼ │      │     =  ρᵢⱼ  =  │
└─ Aᵢⱼ─  Aᵢⱼ₊₁─     └─      L'─
````

ρ=L'*L, return L, where `L`is guaranteed to have positive diagonal elements.
L = cholesky!(ρ).U
If ρ is not exactly positive definite, cholesky will fail
"""
function getL!(A, L; kwargs...)
    Ni,Nj = size(A)
    @inbounds for j = 1:Nj, i = 1:Ni
        # _, ρs, info = eigsolve(ρ->ρmap(ρ,A[i,:],j), L[i,j]'*L[i,j], 1, :LM; ishermitian = false, maxiter = 1, kwargs...)
        # info.converged == 0 && @warn "getL not converged"
        # ρ = real(ρs[1] + ρs[1]')
        _, ρ = simple_eig(ρ->ρmap(ρ,A[i,:],j), L[i,j]'*L[i,j]; kwargs...)
        ρ = real(ρ + ρ') 
        ρ ./= tr(ρ)
        F = svd!(ρ)
        Lo = Diagonal(sqrt.(F.S)) * F.Vt
        _, R = qrpos!(Lo)
        L[i,j] = R
    end
    return L
end

"""
    getAL(A,L)

Given an MPS tensor `A` and `L` ，return a left-canonical MPS tensor `AL`, a gauge transform `R` and
a scalar factor `λ` such that ``λ AR R = L A``
"""
function getAL(A, L)
    Ni,Nj = size(A)
    AL = similar(A)
    Le = similar(L)
    λ = zeros(Ni,Nj)
    @inbounds @views for j = 1:Nj, i = 1:Ni
        Q, R = qrpos!(_to_tail(L[i,j]*_to_front(A[i,j])))
        AL[i,j] = reshape(Q, size(A[i,j]))
        λ[i,j] = norm(R)
        Le[i,j] = rmul!(R, 1/λ[i,j])
    end
    return AL, Le, λ
end

function getLsped(Le, A, AL; kwargs...)
    Ni,Nj = size(A)
    L = similar(Le)
    @inbounds @views for j = 1:Nj, i = 1:Ni
        # λs, Ls, info = eigsolve(X -> ρmap(X,A[i,j],conj(AL[i,j])), Le[i,j], 1, :LM; ishermitian = false, kwargs...)
        # @debug "getLsped eigsolve" λs info sort(abs.(λs))
        # info.converged == 0 && @warn "getLsped not converged"
        # _, Ls1 = selectpos(λs, Ls, Nj)
        _, Ls1 = simple_eig(X -> ρmap(X,A[i,j],conj(AL[i,j])), Le[i,j]; kwargs...)
        _, R = qrpos!(Ls1)
        L[i,j] = R
    end
    return L
end

"""
    left_canonical(A,L=cellones(size(A,1),size(A,2),size(A[1,1],1)); tol = 1e-12, maxiter = 100, kwargs...)

Given an MPS tensor `A`, return a left-canonical MPS tensor `AL`, a gauge transform `L` and
a scalar factor `λ` such that ``λ AL L = L A``, where an initial guess for `L` can be
provided.
"""
function left_canonical(A, L; tol = 1e-12, maxiter = 100, kwargs...)
    # L = getL!(A,L; kwargs...) # seems not necessary
    AL, Le, λ = getAL(A,L;kwargs...)
    numiter = 1
    while norm(L.-Le) > tol && numiter < maxiter
        L = getLsped(Le, A, AL; kwargs...)
        AL, Le, λ = getAL(A, L; kwargs...)
        numiter += 1
    end
    L = Le
    return AL, L, λ
end

"""
    right_canonical(A,L=cellones(size(A,1),size(A,2),size(A[1,1],1)); tol = 1e-12, maxiter = 100, kwargs...)

Given an MPS tensor `A`, return a gauge transform R, a right-canonical MPS tensor `AR`, and
a scalar factor `λ` such that ``λ R AR^s = A^s R``, where an initial guess for `R` can be
provided.
"""
function right_canonical(A, L; tol = 1e-12, maxiter = 100, kwargs...)
    Ni,Nj = size(A)
    Ar = similar(A)
    Lr = similar(L)
    @inbounds for j = 1:Nj, i = 1:Ni
        Ar[i,j] = permute_fronttail(A[i,j])
        Lr[i,j] = permutedims(L[i,j],(2,1))
    end
    AL, L, λ = left_canonical(Ar,Lr; tol = tol, maxiter = maxiter, kwargs...)
    R  = similar(L)
    AR = similar(AL)
    @inbounds for j = 1:Nj, i = 1:Ni
         R[i,j] = permutedims(L[i,j],(2,1))
        AR[i,j] = permute_fronttail(AL[i,j])
    end
    return R, AR, λ
end

"""
    LRtoC(L,R)

```
 ── Cᵢⱼ ──  =  ── Lᵢⱼ ── Rᵢⱼ₊₁ ──
```
"""
function LRtoC(L, R)
    Rijr = circshift(R, (0,-1))
    return [L * R for (L, R) in zip(L, Rijr)]
end

"""
    FLm = FLmap(ALu, ALd, M, FL)

```
  ┌──       ┌──  ALuᵢⱼ  ──                     a ────┬──── c 
  │         │     │                            │     b     │ 
FLᵢⱼ₊₁ =   FLᵢⱼ ─ Mᵢⱼ   ──                     ├─ d ─┼─ e ─┤ 
  │         │     │                            │     g     │ 
  └──       └──  ALdᵢᵣⱼ  ─                     f ────┴──── h 
```
"""

FLmap(FL, ALu, ALd, M::leg4) = ein"((adf,fgh),dgeb),abc -> ceh"(FL, ALd, M, ALu)
FLmap(FL, ALu, ALd, M::leg5) = ein"(((aefi,ijkl),ejgbp),fkhcp),abcd -> dghl"(FL, ALd, M, conj(M), ALu)
FLmap(FL, ALu, ALd, M::leg8) = ein"((aefi,ijkl),efjkghbc),abcd -> dghl"(FL, ALd, M, ALu)

function FLmap(J::Int, FLij, ALui, ALdir, Mi)
    Nj = length(ALui)
    for j in J:(J + Nj - 1)
        jr = mod1(j, Nj)
        FLij = FLmap(FLij, ALui[jr], ALdir[jr], Mi[jr])
    end
    return FLij
end
"""
    FRm = FRmap(ARu, ARd, M, FR, i)

```
    ── ARuᵢⱼ  ──┐          ──┐          a ────┬──── c 
        │       │            │          │     b     │ 
    ── Mᵢⱼ   ──FRᵢⱼ  =    ──FRᵢⱼ₋₁      ├─ d ─┼─ e ─┤ 
        │       │            │          │     g     │ 
    ── ARdᵢᵣⱼ ──┘          ──┘          f ────┴──── h 
```
"""
FRmap(FR, ARu, ARd, M::leg4) = ein"((abc,ceh),dgeb),fgh -> adf"(ARu, FR, M, ARd)
FRmap(FR, ARu, ARd, M::leg5) = ein"(((abcd,dghl),ejgbp),fkhcp),ijkl -> aefi"(ARu, FR, M, conj(M), ARd)
FRmap(FR, ARu, ARd, M::leg8) = ein"((abcd,dghl),efjkghbc),ijkl -> aefi"(ARu, FR, M, ARd)

function FRmap(J::Int, FRij, ARui, ARdir, Mi)
    Nj = length(ARui)
    for j in J:-1:(J - Nj + 1)
        jr = mod1(j, Nj)
        FRij = FRmap(FRij, ARui[jr], ARdir[jr], Mi[jr])
    end
    return FRij
end

function FLint(AL, M::leg4; kwargs...)
    Ni, Nj = size(AL)
    χ = size(AL[1], 1)
    atype = _arraytype(AL[1])
    return [(D = size(M[i, j], 1); atype(rand(ComplexF64, χ, D, χ))) for i = 1:Ni, j = 1:Nj]
end

function FLint(AL, M::leg5; kwargs...)
    Ni, Nj = size(AL)
    χ = size(AL[1], 1)
    atype = _arraytype(AL[1])
    return [(D = size(M[i, j], 1); atype(rand(ComplexF64, χ, D, D, χ))) for i = 1:Ni, j = 1:Nj]
end

function FLint(AL, M::leg8; kwargs...)
    Ni, Nj = size(AL)
    χ = size(AL[1], 1)
    atype = _arraytype(AL[1])
    return [(D = size(M[i, j], 1); atype(rand(ComplexF64, χ, D, D, χ))) for i = 1:Ni, j = 1:Nj]
end

function FLint(AL, M::Matrix{<:U1Array}; alg)
    Ni,Nj = size(AL)
    χ = size(AL[1],1)
    atype = _arraytype(AL[1])
    FL = Array{atype{ComplexF64, 4}, 2}(undef, Ni, Nj)
    qnD, qnχ, dimsD, dimsχ = alg.U1info
    dir = [1, getdir(M[1])[1], -getdir(M[1])[1], -1]
    indqn = [qnχ, qnD, qnD, qnχ]
    indims = [dimsχ, dimsD, dimsD, dimsχ]
    for j in 1:Nj, i in 1:Ni
        D = size(M[i,j], 1)
        FL[i,j] = randinitial(AL[i,j], χ, D, D, χ; 
                              dir = dir, indqn = indqn, indims = indims
        )
    end
    return FL
end

function FRint(AR, M::leg4; kwargs...)
    Ni, Nj = size(AR)
    χ = size(AR[1], 1)
    atype = _arraytype(AR[1])
    return [(D = size(M[i, j], 3); atype(rand(ComplexF64, χ, D, χ))) for i = 1:Ni, j = 1:Nj]
end

function FRint(AR, M::leg5; kwargs...)
    Ni, Nj = size(AR)
    χ = size(AR[1], 1)
    atype = _arraytype(AR[1])
    return [(D = size(M[i, j], 3); atype(rand(ComplexF64, χ, D, D, χ))) for i = 1:Ni, j = 1:Nj]
end

function FRint(AR, M::leg8; kwargs...)
    Ni, Nj = size(AR)
    χ = size(AR[1], 1)
    atype = _arraytype(AR[1])
    return [(D = size(M[i, j], 5); atype(rand(ComplexF64, χ, D, D, χ))) for i = 1:Ni, j = 1:Nj]
end

function FRint(AR, M::Matrix{<:U1Array}; alg)
    Ni,Nj = size(AR)
    χ = size(AR[1], 4)
    atype = _arraytype(AR[1])
    FR = Array{atype{ComplexF64, 4}, 2}(undef, Ni, Nj)
    qnD, qnχ, dimsD, dimsχ = alg.U1info
    dir = [-1, getdir(M[1])[3], -getdir(M[1])[3], 1]
    indqn = [qnχ, qnD, qnD, qnχ]
    indims = [dimsχ, dimsD, dimsD, dimsχ]
    for j in 1:Nj, i in 1:Ni
        D = size(M[i,j], 5)
        FR[i,j] = randinitial(AR[i,j], χ, D, D, χ; 
                              dir = dir, indqn = indqn, indims = indims
        )
    end

    return FR
end

"""
    λL, FL = leftenv(ALu, ALd, M, FL = FLint(ALu,M); kwargs...)

Compute the left environment tensor for MPS A and MPO M, by finding the left fixed point
of ALu - M - ALd contracted along the physical dimension.
```
 ┌──  ALuᵢⱼ  ──          ┌── 
 │     │                 │   
FLᵢⱼ ─ Mᵢⱼ   ──   = λLᵢⱼ FLᵢⱼ₊₁   
 │     │                 │   
 └──  ALdᵢᵣⱼ  ─          └── 
```
"""
function leftenv(ALu, ALd, M, FL=FLint(ALu,M); ifobs=false, alg, kwargs...) 
    Ni, Nj = size(M)
    λL = Zygote.Buffer(zeros(ComplexF64, Ni))
    FL′ = Zygote.Buffer(FL)
    for i in 1:Ni
        ir = ifobs ? Ni + 1 - i : mod1(i + 1, Ni)
        if alg.ifsimple_eig
            if alg.ifcheckpoint
                λL[i], FL′[i,1] = checkpoint(simple_eig, FLij -> FLmap(1, FLij, ALu[i,:], ALd[ir,:], M[i, :]), FL[i,1])
            else
                λL[i], FL′[i,1] = simple_eig(FLij -> FLmap(1, FLij, ALu[i,:], ALd[ir,:], M[i, :]), FL[i,1])
            end
        else
            λLs, FLi1s, info = eigsolve(FLij -> FLmap(1, FLij, ALu[i,:], ALd[ir,:], M[i, :]), 
                                        FL[i,1], 1, :LM; alg_rrule=GMRES(verbosity=-1), maxiter=100, ishermitian=false, kwargs...)
            alg.verbosity >= 1 && info.converged == 0 && @warn "leftenv not converged"
            λL[i], FL′[i,1] = selectpos(λLs, FLi1s, Nj)
        end
        for j in 2:Nj
            FL′[i,j] = FLmap(FL′[i,j-1], ALu[i,j-1], ALd[ir,j-1],  M[i,j-1])
        end
    end
    
    return copy(λL), copy(FL′)
end

"""
    λR, FR = rightenv(ARu, ARd, M, FR = FRint(ARu,M); kwargs...)

Compute the right environment tensor for MPS A and MPO M, by finding the left fixed point
of AR - M - conj(AR) contracted along the physical dimension.
```
    ── ARuᵢⱼ  ──┐          ──┐   
        │       │            │  
    ── Mᵢⱼ   ──FRᵢⱼ  = λRᵢⱼ──FRᵢⱼ₋₁
        │       │            │  
    ── ARdᵢᵣⱼ ──┘          ──┘  
```
"""
function rightenv(ARu, ARd, M, FR=FRint(ARu,M); ifobs=false, alg, kwargs...) 
    Ni,Nj = size(M)
    λR = Zygote.Buffer(zeros(ComplexF64, Ni))
    FR′ = Zygote.Buffer(FR)
    for i in 1:Ni
        ir = ifobs ? Ni + 1 - i : mod1(i + 1, Ni)
        if alg.ifsimple_eig
            if alg.ifcheckpoint
                λR[i], FR′[i,Nj] = checkpoint(simple_eig, FRiNj -> FRmap(Nj, FRiNj, ARu[i,:], ARd[ir,:], M[i,:]), FR[i,Nj])
            else
                λR[i], FR′[i,Nj] = simple_eig(FRiNj -> FRmap(Nj, FRiNj, ARu[i,:], ARd[ir,:], M[i,:]), FR[i,Nj])
            end
        else
            λRs, FR1s, info = eigsolve(FRiNj -> FRmap(Nj, FRiNj, ARu[i,:], ARd[ir,:], M[i,:]), 
                                    FR[i,Nj], 1, :LM; alg_rrule=GMRES(verbosity=-1), maxiter=100, ishermitian = false, kwargs...)
            alg.verbosity >= 1 && info.converged == 0 && @warn "rightenv not converged"
            λR[i], FR′[i,Nj] = selectpos(λRs, FR1s, Nj)
        end
        for j in Nj-1:-1:1
            FR′[i,j] = FRmap(FR′[i,j+1], ARu[i,j+1], ARd[ir,j+1], M[i,j+1])
        end
    end
    return copy(λR), copy(FR′)
end

"""
    Rm = Rmap(FRi::Vector{<:AbstractTensorMap}, 
                ARui::Vector{<:AbstractTensorMap}, 
                ARdir::Vector{<:AbstractTensorMap}, 
                )

```
    ── ARuᵢⱼ  ──┐          ──┐           a──────┬──────c    
        │       Rᵢⱼ  =       Rᵢⱼ₋₁       │      │      │ 
    ── ARdᵢᵣⱼ ──┘          ──┘           │      b      │    
                                         │      │      │      
                                         d──────┴──────e   
```
"""
function Rmap(Ri, ARui, ARdir)
    Rm = [ein"(abc,ce),dbe->ad"(ARu, R, ARd) for (R, ARu, ARd) in zip(Ri, ARui, ARdir)]
    return circshift(Rm, -1)
end

"""
    λR, FR = rightCenv(ARu::Matrix{<:AbstractTensorMap}, 
                       ARd::Matrix{<:AbstractTensorMap}, 
                       R::Matrix{<:AbstractTensorMap} = initial_C(ARu); 
                       kwargs...) 

Compute the right environment tensor for MPS A by finding the left fixed point
of AR - conj(AR) contracted along the physical dimension.
```
    ── ARuᵢⱼ  ──┐          ──┐   
        |       Rᵢⱼ  = λRᵢⱼ  Rᵢⱼ₋₁
    ── ARdᵢᵣⱼ ──┘          ──┘  
```
"""
function rightCenv(ARu, ARd, R=cellones(ARu); 
                   ifobs=false, verbosity=Defaults.verbosity, kwargs...) 

    Ni, Nj = size(ARu)
    λR = Zygote.Buffer(zeros(eltype(ARu[1]), Ni))
    R′ = Zygote.Buffer(R)
    for i in 1:Ni
        ir = ifobs ? mod1(Ni - i + 2, Ni) : i
        λRs, R1s, info = eigsolve(R -> Rmap(R, ARu[i,:], ARd[ir,:]), R[i,:], 1, :LM; 
                                  alg_rrule=GMRES(verbosity=-1), maxiter=100, ishermitian = false, kwargs...)
        verbosity >= 1 && info.converged == 0 && @warn "rightenv not converged"
        λR[i], R′[i,:] = selectpos(λRs, R1s, Nj)
    end
    return copy(λR), copy(R′)
end

"""
    ACm = ACmap(ACij, FLj, FRj, Mj, II)

```
                                ┌─────── ACᵢⱼ ─────┐              a ────┬──── c  
┌───── ACᵢ₊₁ⱼ ─────┐            │        │         │              │     b     │ 
│        │         │      =     FLᵢⱼ ─── Mᵢⱼ ───── FRᵢⱼ           ├─ d ─┼─ e ─┤ 
                                │        │         │              │     g     │ 
                                                                  f ────┴──── h 
                                                               
```
"""
ACmap(AC, FL, FR, M::leg4) = ein"((abc,ceh),dgeb),adf -> fgh"(AC,FR,M,FL)
ACmap(AC, FL, FR, M::leg5) = ein"(((abcd,dghl),ejgbp),fkhcp),aefi -> ijkl"(AC,FR,M,conj(M),FL)
ACmap(AC, FL, FR, M::leg8) = ein"((abcd,dghl),efjkghbc),aefi -> ijkl"(AC,FR,M,FL)

function ACmap(I::Int, ACij, FLj, FRj, Mj)
    Ni = length(FLj)
    for i in I:(I + Ni - 1)
        ir = mod1(i, Ni)
        ACij = ACmap(ACij, FLj[ir], FRj[ir], Mj[ir])
    end
    return ACij
end
"""
    Cmap(Cij, FLjp, FRj, II)

```
                    ┌────Cᵢⱼ ───┐            a ─── b
┌── Cᵢ₊₁ⱼ ──┐       │           │            │     │
│           │  =   FLᵢⱼ₊₁ ──── FRᵢⱼ          ├─ c ─┤
                    │           │            │     │
                                             d ─── e                                    
```
"""
Cmap(C, FL::leg3, FR) = ein"acd,(ab,bce) -> de"(FL,C,FR)
Cmap(C, FL::leg4, FR) = ein"acde,(ab,bcdf) -> ef"(FL,C,FR)

function Cmap(I, Cij, FLjr, FRj)
    Ni = length(FLjr)
    for i in I:(I + Ni - 1)
        ir = mod1(i, Ni)
        Cij = Cmap(Cij, FLjr[ir], FRj[ir])
    end
    return Cij
end

"""
    ACenv(AC, FL, M, FR;kwargs...)

Compute the up environment tensor for MPS `FL`,`FR` and MPO `M`, by finding the up fixed point
        of `FL - M - FR` contracted along the physical dimension.
```
┌─────── ACᵢⱼ ─────┐         
│        │         │         =  λACᵢⱼ ┌─── ACᵢ₊₁ⱼ ──┐
FLᵢⱼ ─── Mᵢⱼ ───── FRᵢⱼ               │      │      │   
│        │         │   
```
"""
function ACenv(AC, FL, M, FR; alg, kwargs...)
    Ni, Nj = size(M)
    λAC = Zygote.Buffer(zeros(ComplexF64, Nj))
    AC′ = Zygote.Buffer(AC)
    for j in 1:Nj
        if alg.ifsimple_eig
            if alg.ifcheckpoint
                λAC[j], AC′[1,j] = checkpoint(simple_eig, AC1j -> ACmap(1, AC1j, FL[:,j], FR[:,j], M[:,j]), AC[1,j])
            else
                λAC[j], AC′[1,j] = simple_eig(AC1j -> ACmap(1, AC1j, FL[:,j], FR[:,j], M[:,j]), AC[1,j])
            end
        else
            λACs, ACs, info = eigsolve(AC1j -> ACmap(1, AC1j, FL[:,j], FR[:,j], M[:,j]), 
                                    AC[1,j], 1, :LM; alg_rrule=GMRES(verbosity=-1), maxiter=100, ishermitian = false, kwargs...)
            alg.verbosity >= 1 && info.converged == 0 && @warn "ACenv Not converged"
            λAC[j], AC′[1,j] = selectpos(λACs, ACs, Ni)
        end
        for i in 2:Ni
            AC′[i,j] = ACmap(AC′[i-1,j], FL[i-1,j], FR[i-1,j], M[i-1,j])
        end
    end
    return copy(λAC), copy(AC′)
end

"""
    Cenv(C, FL, FR;kwargs...)

Compute the up environment tensor for MPS `FL` and `FR`, by finding the up fixed point
    of `FL - FR` contracted along the physical dimension.
```
┌────Cᵢⱼ ───┐
│           │       =  λCᵢⱼ ┌──Cᵢⱼ ─┐
FLᵢⱼ₊₁ ──── FRᵢⱼ            │       │
│           │   
```
"""
function Cenv(C, FL, FR; alg, kwargs...)
    Ni, Nj = size(C)
    λC = Zygote.Buffer(zeros(ComplexF64, Nj))
    C′ = Zygote.Buffer(C)
    for j in 1:Nj
        jr = mod1(j + 1, Nj)
        if alg.ifsimple_eig
            if alg.ifcheckpoint
                λC[j], C′[1,j] = checkpoint(simple_eig, C1j -> Cmap(1, C1j, FL[:,jr], FR[:,j]), C[1,j])
            else
                λC[j], C′[1,j] = simple_eig(C1j -> Cmap(1, C1j, FL[:,jr], FR[:,j]), C[1,j])
            end
        else
            λCs, Cs, info = eigsolve(C1j -> Cmap(1, C1j, FL[:,jr], FR[:,j]), 
                                    C[1,j], 1, :LM; alg_rrule=GMRES(verbosity=-1), maxiter=100, ishermitian = false, kwargs...)
            alg.verbosity >= 1 && info.converged == 0 && @warn "Cenv Not converged"
            λC[j], C′[1,j] = selectpos(λCs, Cs, Ni)
        end
        for i in 2:Ni
            C′[i,j] = Cmap(C′[i-1,j], FL[i-1,jr], FR[i-1,j])
        end
    end
    return copy(λC), copy(C′)
end

ALCtoAC(AL::leg3, C) = [ein"asc,cb -> asb"(AL, C) for (AL, C) in zip(AL, C)]
ALCtoAC(AL::leg4, C) = [ein"astc,cb -> astb"(AL, C) for (AL, C) in zip(AL, C)]

function ACCtoAL(ACij, Cij)
    QAC, RAC = qrpos(_to_tail(ACij))
     QC, RC  = qrpos(Cij)
    errL = norm(RAC-RC)
    AL = reshape(QAC*QC', size(ACij))
    return AL, errL
end

function ACCtoAR(ACij, Cijr)
    LAC, QAC = lqpos(_to_front(ACij))
     LC, QC  = lqpos(Cijr)
    errR = norm(LAC-LC)
    AR = reshape(QC'*QAC, size(ACij))
    return AR, errR
end


"""
    AL, AR = ACCtoALAR(AC, C)

QR factorization to get `AL` and `AR` from `AC` and `C`

````
──ALᵢⱼ──Cᵢⱼ──  =  ──ACᵢⱼ──  = ──Cᵢ₋₁ⱼ ──ARᵢⱼ──
  │                  │                  │   
````
"""
function ACCtoALAR(AC, C)
    AC = env_norm(AC)
     C = env_norm( C)
    ALijerrL = [ACCtoAL(AC, C) for (AC, C) in zip(AC, C)]
    AL = [ALij for (ALij, _) in ALijerrL]
    errL = Zygote.@ignore sum([errL for (_, errL) in ALijerrL])
    Ni, Nj = size(AC)
    ARijerrR = [ACCtoAR(AC[i,j], C[i,mod1(j-1,Nj)]) for i=1:Ni, j in 1:Nj]
    AR = [ARij for (ARij, _) in ARijerrR]
    errR = Zygote.@ignore sum([errR for (_, errR) in ARijerrR])
    return AL, AR, errL, errR
end