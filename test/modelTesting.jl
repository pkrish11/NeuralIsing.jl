# Test the functions in the ModelTesting.jl file

@testset "model_testing utilities" begin
    @testset "energy function" begin

        ham = NeuralIsing.IsingHamiltonian(1,0)
        sys = ones(Int, 2,2)

        @test NeuralIsing.calculate_total_Energy(sys,ham) == -8

        ham2 = NeuralIsing.IsingHamiltonian(0,1)
        @test NeuralIsing.calculate_total_Energy(sys,ham2) == -4

        sys_large = sign.( 2 .* rand(10,10) .- 1) # get random system
        energy = 0
        J, h = 1,1
        for i in 1:10  # brute force computation 
            for j in 1:10
                energy += -J * (sys_large[i,j] * sys_large[i,mod1(j+1,10)] + sys_large[i,j] * sys_large[mod1(i+1,10), j]) - h*sys_large[mod1(i+1,10), j]
            end
        end
        ham_test =  NeuralIsing.IsingHamiltonian(1,1)
        @test energy == NeuralIsing.calculate_total_Energy(sys_large, ham_test)  
    end

    
    @testset "neighbour updates" begin

    # test if compute-sub-system function finds the correct neighbours according to PBCs, on a 3x3 system
    test_sys = ones(Int, 3,3)
    NNFlip = NeuralIsing.NNFlip(2, 3) # make 2x2 update on 3x3 grid
    FlipInfo = NeuralIsing.compute_sub_system(CartesianIndex(3,3), NNFlip, test_sys)
    
    true_sub_sys = [CartesianIndex(3,3), CartesianIndex(1,3), CartesianIndex(3,1), CartesianIndex(1,1)]
    @test Set(FlipInfo.sys_indices) == Set(true_sub_sys)

    # test if the update_system function acts on correct indices in the larger system
    updated_test_sys = -1 * ones(Int, 2,2)
    NeuralIsing.update_system!(updated_test_sys, FlipInfo, test_sys)

    @test Set(test_sys[true_sub_sys]) == Set(updated_test_sys)
    
    end

end
