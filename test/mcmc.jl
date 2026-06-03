# Test functions in the mcmc.jl file, used to run single-spin flip simulations of 2D Ising model and generate data.

@testset "Single Spin Flip MCMC" begin
    @testset "Neighbours, Energy, Magnetization functions" begin
        lattice = NeuralIsing.twoDLattice(3) 
        system =  NeuralIsing.SpinSystem(lattice, initial_state=:random) 
        ham =  NeuralIsing.IsingHamiltonian(1., 1.)

        # check if neighbours of (1,3) position are correct, and if neighbours! function updates system correctly.
        site = 3
        NeuralIsing.neighbors!(system.lattice, site)
        @test Set(system.lattice.neighbors) == Set([1, 9, 2, 6])

        # check if total energy and ΔE functions are coherent.
        move_info =  NeuralIsing.propose_move(system, ham, NeuralIsing.SingleSpinFlip())
        ΔE_direct = NeuralIsing.calculate_ΔE(system, move_info, ham, NeuralIsing.SingleSpinFlip())
        updated_sys = copy(system)
        updated_sys.spins[move_info.index] *= -1
        ΔE = NeuralIsing.total_energy(updated_sys, ham) - NeuralIsing.total_energy(system, ham)

        @test ΔE_direct == ΔE

        # test if magnetization is computed correctly
        lattice2 = NeuralIsing.twoDLattice(3) 
        ones_system =  NeuralIsing.SpinSystem(lattice2) 
        ones_system.spins[:] = [1,1,1,1,-1,-1,-1,-1,1]

        @test NeuralIsing.compute_magnetization(ones_system) == (1/9)

    end
end

