push!(LOAD_PATH,"../src/")
using Documenter, ThymusABM

makedocs(sitename="My Documentation",
	authors = ["Ethan Mulle", "Hannah Meyer"],
	pages = [
		"Home" => "index.md",
		"Introduction" => "introduction.md",
		"Instructions" => "instructions.md"
		]
	)

deploydocs(repo = "github.com/meyer-lab-cshl/in-silico-thymus.git")
