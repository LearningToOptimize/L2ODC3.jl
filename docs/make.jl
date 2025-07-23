using Documenter
using L2ODC3


include("definitions.jl")

makedocs(
    modules=[L2ODC3],
    sitename = "L2ODC3.jl",
    format = Documenter.HTML(;
        assets = ["assets/wider.css", "assets/redlinks.css"],
        mathengine = Documenter.MathJax3(Dict(
            :tex => Dict(
                "macros" => make_macros_dict("docs/src/assets/definitions.tex"),
                "inlineMath" => [["\$","\$"], ["\\(","\\)"]],
                "tags" => "ams",
            ),
        )),
    ),
    pages = [
        "Home" => "index.md",
    ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo="github.com/LearningToOptimize/L2ODC3.jl.git",
    push_preview=true,
)