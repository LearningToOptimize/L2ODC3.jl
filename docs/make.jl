using Documenter
using DocumenterMermaid
using Literate
using L2OCore
using L2ODC3

include("definitions.jl")

# Literate examples: docs/src/examples/*.jl -> docs/src/examples/*.md (executed by Documenter).
const EXAMPLES = joinpath(@__DIR__, "src", "examples")
for file in readdir(EXAMPLES)
    endswith(file, ".jl") || continue
    Literate.markdown(joinpath(EXAMPLES, file), EXAMPLES; documenter=true, credit=false)
end

DocMeta.setdocmeta!(L2ODC3, :DocTestSetup, :(using L2ODC3, L2OCore); recursive=true)

makedocs(;
    modules=[L2ODC3],
    sitename="L2ODC3.jl",
    authors="LearningToOptimize.jl contributors",
    doctest=true,
    checkdocs=:exports,
    warnonly=false,
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://LearningToOptimize.github.io/L2ODC3.jl",
        assets=["assets/wider.css", "assets/redlinks.css"],
        mathengine=Documenter.MathJax3(
            Dict(
                :tex => Dict(
                    "macros" => make_macros_dict(joinpath(@__DIR__, "src", "assets", "definitions.tex")),
                    "inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
                    "tags" => "ams",
                ),
            ),
        ),
    ),
    pages=[
        "Home" => "index.md",
        "Theory" => "theory.md",
        "Package guide" => "guide.md",
        "Examples" => ["Training on the DC3 paper's QP" => "examples/dc3_qp.md"],
        "API reference" => "api.md",
    ],
)

deploydocs(; repo="github.com/LearningToOptimize/L2ODC3.jl.git", devbranch="main", push_preview=true)
