module NeuralIsing

using DelimitedFiles
using Revise
using StatsBase

include("mcmc.jl")
include("analyseData.jl")

include("ML.jl")
include("ModelTesting.jl")

end 
