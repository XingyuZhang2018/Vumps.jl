module TeneT

using ChainRulesCore
using CUDA
using LinearAlgebra
using KrylovKit
using OMEinsum
using Printf
using Parameters
using U1ArrayKit
using Zygote

import Base: +, -, *, getindex, Array
import LinearAlgebra: norm,  mul!
import VectorInterface: inner, scale, scale!!, scalartype, zerovector, add!!
import CUDA: CuArray
# import KrylovKit: RealVec
export VUMPS, VUMPSRuntime, VUMPSEnv
export leading_boundary


include("defaults.jl")
include("utilities.jl")
include("patch.jl")
include("environment.jl")
include("vumpsruntime.jl")
include("grassmann.jl")
include("autodiff.jl")


end
