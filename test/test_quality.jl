@testset "Aqua" begin
    Aqua.test_all(L2ODC3)
end

@testset "ExplicitImports" begin
    @test ExplicitImports.check_no_implicit_imports(L2ODC3) === nothing
    @test ExplicitImports.check_no_stale_explicit_imports(L2ODC3) === nothing
    @test ExplicitImports.check_all_qualified_accesses_via_owners(L2ODC3) === nothing
    @test ExplicitImports.check_no_self_qualified_accesses(L2ODC3) === nothing
end

@testset "JET" begin
    if pkgversion(JET) >= v"0.10"
        JET.test_package(L2ODC3; target_modules=(L2ODC3,))
    else
        JET.test_package(L2ODC3; target_defined_modules=true)
    end
end
