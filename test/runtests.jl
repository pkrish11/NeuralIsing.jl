using NeuralIsing
using Test, Random, Statistics 
 
# Executing Pkg.test() in the REPL will run all the tests and return information about pass/fail.


function findTests()
    allFiles = readdir(@__DIR__)
    testFiles = filter(f -> endswith(f, ".jl") && f != "runtests.jl", allFiles)
    return [replace(f, ".jl" => "") for f in testFiles]
end

const allTests = findTests()
const testsToRun = isempty(ARGS) ? allTests : ARGS

@testset "NeuralIsing.jl" begin
    for test in testsToRun
        include("$(test).jl")
    end
end
