using L2ODC3
using L2OCore
using Test

using Adapt: Adapt
using Aqua: Aqua
using ExplicitImports: ExplicitImports
using FiniteDifferences: FiniteDifferences
using ForwardDiff: ForwardDiff
using HiGHS: HiGHS
using Ipopt: Ipopt
using JET: JET
using Zygote: Zygote
using JLArrays: JLArray

include("problems.jl")
include("properties.jl")

@testset "L2ODC3" begin
    @testset "problem forms" include("test_forms.jl")
    @testset "equality completion" include("test_completion.jl")
    @testset "correction and feasibility seeking" include("test_correction.jl")
    @testset "methods and pipeline" include("test_methods.jl")
    @testset "solver optima" include("test_optima.jl")
    @testset "reference implementations" include("test_reference.jl")
    @testset "GPU array types" include("test_gpu.jl")
    @testset "code quality" include("test_quality.jl")
end
