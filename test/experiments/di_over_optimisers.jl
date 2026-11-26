using Test
using Optimisers

using ForwardDiff
using Zygote
using Enzyme
using Mooncake

using DifferentiationInterface
const DI = DifferentiationInterface

@testset "DI over Optimisers" begin
    function test_nested(outer, inner)
        function run_adam_steps(N, x0)
            x = copy(x0)
            rule = Optimisers.Adam(0.1)
            state = Optimisers.setup(rule, x)
            for _ in 1:N
                grad = DI.gradient(x -> sum(abs2.(x)), inner(), x)
                state, x = Optimisers.update(state, x, grad)
            end
            return sum(x)
        end

        x0 = [1.0, 2.0, 3.0]
        N = 5
        run_adam_steps(N, x0)

        grad = DI.gradient(x -> run_adam_steps(N, x), outer(), x0)

        @test length(grad) == length(x0)
        @test all(!isnan, grad)
        @test all(>(0), grad)

        return true
    end
    backends = [AutoForwardDiff, AutoZygote, AutoEnzyme, AutoMooncake]
    working_combos = Dict(k => Set() for k in backends)
    for _outer in backends
        for _inner in backends
            @testset "$_outer over $_inner" begin
                if (
                    (_outer in (AutoZygote, AutoEnzyme, AutoMooncake)) ||
                    (_outer === AutoForwardDiff && _inner in (AutoEnzyme, AutoMooncake))
                )
                    @test_throws "" test_nested(_outer, _inner)
                else
                    @test test_nested(_outer, _inner)
                    push!(working_combos[_outer], _inner)
                end
            end
        end
    end
    ds = "Working combinations:\n"
    for _outer in keys(working_combos)
        ds *= "$_outer over: "
        ds *= join(collect(working_combos[_outer]), ", ")
        ds *= "\n"
    end
    @info ds
end
