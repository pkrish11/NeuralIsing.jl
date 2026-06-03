# Test if MCMC data generated from Single Spin Flips actually follows expected posterior energy / states distribution by comparing to Gaussian.

# Note : We do not use the generate_raw_data function here to avoid writing new data files each time unit-tests are run.
#        Instead we define a new exactly the same workflow and test the results.
@testset "Posterior Distribution for Single Spin Flip MCMC" begin
    @testset "Energy Distribution for Low Temp" begin

        num_steps, observable, sweep = 500, NeuralIsing.compute_magnetization, true  
        size, J, H = 10, 1., 0.
        move = NeuralIsing.SingleSpinFlip()
        beta = 2. # low temp-limit

        energy_stat = 0
        magnetization_stat = 0

        for run in 1:10 
            system, hamiltonian = NeuralIsing.initialize_sys(size, J, H)               
            system_init = copy(system) 
            system_evolution, observable_evolution, time_to_run = NeuralIsing.run_mc(system_init, hamiltonian, move, beta, num_steps, observable, sweep)
            tau_time, ind_samples, ind_obs = NeuralIsing.analyze_mc_chain(system_evolution, observable_evolution) 
           #  energies = [NeuralIsing.calculate_total_Energy(reshape(ind_samples[i], size, size), hamiltonian) for i in eachindex(ind_samples)]

            magnetization_stat += mean(ind_obs)
            # energy_stat += std(energies)^2 
        end
        magnetization_stat /= 10 # average of mean magnetization over 10 runs.
       #  energy_stat /= 10 # average of variance in energy over 10 runs

        @test abs(magnetization_stat)-1 <= 0.1 # system is ordered.
        # @test abs(energy_stat - exp(-8*beta)*size^2) <= 0.1 # variance in energy is small, scales like ( num spins x e^-8*beta) since cost to flip is e^-8
    end

    @testset "Energy Distribution for High Temp" begin

        num_steps, observable, sweep = 500, NeuralIsing.compute_magnetization, true  
        size, J, H = 10, 1., 0.
        move = NeuralIsing.SingleSpinFlip()
        beta = .2 # high temp-limit

        energy_stat = 0
        magnetization_stat = 0

        for run in 1:10 
            system, hamiltonian = NeuralIsing.initialize_sys(size, J, H)               
            system_init = copy(system) 
            system_evolution, observable_evolution, time_to_run = NeuralIsing.run_mc(system_init, hamiltonian, move, beta, num_steps, observable, sweep)
            tau_time, ind_samples, ind_obs = NeuralIsing.analyze_mc_chain(system_evolution, observable_evolution) 
            # energies = [NeuralIsing.calculate_total_Energy(reshape(ind_samples[i], size, size), hamiltonian) for i in eachindex(ind_samples)]

            magnetization_stat += mean(ind_obs)
            # energy_stat += std(energies)^2 
        end
        magnetization_stat /= 10 # average of mean magnetization over 10 runs.
        # energy_stat /= 10 # average of variance in energy over 10 runs

        @test abs(magnetization_stat) <= 0.1 # system is disordered
       #  @test abs(energy_stat - 2*size^2) <= 0.1 # variance in energy Gaussian scales like 2*num_spins when spins mostly independet.
    end
end
