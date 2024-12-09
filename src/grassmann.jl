"""
    Project a vector g onto a Grassmann (ij)(k) manifold at point x
    x: shape: (i,j)(k)
    g: shape: (i,j)(k)
"""
project_AL(∂AL, AL) = project_AL!(deepcopy(∂AL), AL)
function project_AL!(∂AL, AL)
    if ∂AL[1] isa InnerProductVec
        # ∂AL = [RealVec(∂AL.vec - ein"deg,(abc,abg)->dec"(AL, ∂AL.vec, conj.(AL))) for (∂AL, AL) in zip(∂AL, AL)]
        # @show 1111111
        ∂AL = [(s = real(ein"(abc,abg)->cg"(∂AL.vec, conj.(AL))); RealVec(ComplexF64.(real(∂AL.vec - real(ein"deg,cg->dec"(AL, s)))))) for (∂AL, AL) in zip(∂AL, AL)]
    else
        [∂AL isa AbstractZero || (∂AL .-= ein"deg,(abc,abg)->dec"(AL, ∂AL, conj.(AL))) for (∂AL, AL) in zip(∂AL, AL)]
    end
    return ∂AL
end

project_AR(∂AR, AR) = project_AR!(deepcopy(∂AR), AR)
function project_AR!(∂AR, AR)
    ∂AL = permute_fronttail.(∂AR)
    AL = permute_fronttail.(AR)
    ∂AR .= permute_fronttail.(project_AL!(∂AL, AL))
    return ∂AR
end

"""
    Retrct a vector g onto a Grassmann manifold at point x.
     This is a SVD based retraction and same as it in the Stiefel manifold.
"""
function retract!(x)
    AL, _, _ = left_canonical(x)
    x .= AL
    return x
end