const leg3 = Union{<:AbstractArray{ComplexF64, 3}, Vector{<:AbstractArray{ComplexF64, 3}}, Matrix{<:AbstractArray{ComplexF64, 3}}}
const leg4 = Union{<:AbstractArray{ComplexF64, 4}, Vector{<:AbstractArray{ComplexF64, 4}}, Matrix{<:AbstractArray{ComplexF64, 4}}}
const leg5 = Union{<:AbstractArray{ComplexF64, 5}, Vector{<:AbstractArray{ComplexF64, 5}}, Matrix{<:AbstractArray{ComplexF64, 5}}}

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

orth_for_ad(v) = v
function simple_eig(f, v; n=20)
    λ = 0.0
    err = 1.0
    i = 0
    while err > 1e-12 && i < n
        Zygote.@ignore begin
            v = f(v)
            v /= norm(v)
            CUDA.@allowscalar λ = Array(f(v)[1:2] ./ v[1:2])
            err = norm(λ[1] - λ[2])
            i += 1
        end
    end
    for _ in 1:5
        v = f(v)
        v /= norm(v)
    end

    v = orth_for_ad(v)

    return λ[1], v
end
