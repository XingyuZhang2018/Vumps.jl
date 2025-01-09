const leg3 = Union{<:AbstractArray{T, 3}, Vector{<:AbstractArray{T, 3}}, Matrix{<:AbstractArray{T, 3}}} where T
const leg4 = Union{<:AbstractArray{T, 4}, Vector{<:AbstractArray{T, 4}}, Matrix{<:AbstractArray{T, 4}}} where T
const leg5 = Union{<:AbstractArray{T, 5}, Vector{<:AbstractArray{T, 5}}, Matrix{<:AbstractArray{T, 5}}} where T
const doublearray = Union{<:DoubleArray, Vector{<:DoubleArray}, Matrix{<:DoubleArray}}

function _to_front(t)
    χ = size(t)[end]
    return reshape(t, χ, Int(prod(size(t))/χ))
end

function _to_tail(t)
    χ = size(t, 1)
    return reshape(t, Int(prod(size(t))/χ), χ)
end

permute_fronttail(t::leg3) = permutedims(t, (3,2,1))
permute_fronttail(t::leg4) = permutedims(t, (4,2,3,1))
permute_fronttail(t::InnerProductVec) = RealVec(permute_fronttail(t.vec))
permute_fronttail(t::AbstractZero) = t

orth_for_ad(v) = v
function simple_eig(f, v; max_iter=100)
    λ_old = 0.0
    Zygote.@ignore begin
        for _ in 1:max_iter
            v = f(v)
            λ = norm(v)
            v /= λ
            # @show abs(λ - λ_old)
            abs(λ - λ_old) < 1e-8 && break
            λ_old = λ
        end
    end
    for _ in 1:5
        v = f(v)
        v /= norm(v)
    end

    v = orth_for_ad(v)
    if v isa DoubleArray
        v′ = f(v)
        CUDA.@allowscalar λ = v′.real.tensor[1] ./ v.real.tensor[1] 
    else
        CUDA.@allowscalar λ = f(v)[1] ./ v[1]
        @show f(v) ./ v
    end
    
    return λ, v
end

function mcform(M)
    aM = Array(M)
    x = ein"ijil->jl"(aM)
    _, vh = Zygote.@ignore eigen(x)
    aM = ein"aj,(ijkl,lb)->iakb"(inv(vh),aM,vh)
    y = ein"ijkj->ik"(aM)
    _, vv = Zygote.@ignore eigen(y)
    aM = ein"(ai,ijkl),kb->ajbl"(inv(vv),aM,vv)
    aM = typeof(M)(aM)
    return vh, vv, aM
end    

# See Zygote Checkpointing https://fluxml.ai/Zygote.jl/latest/adjoints/#Checkpointing-1
checkpoint(f, x...) = f(x...) 
Zygote.@adjoint checkpoint(f, x...) = f(x...), ȳ -> Zygote._pullback(f, x...)[2](ȳ)