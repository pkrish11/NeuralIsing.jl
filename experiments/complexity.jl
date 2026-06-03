using NeuralIsing
using Plots

# compare (wall-clock) time complexity of MCMC (single spin flip) versus model
# at a given beta value, for varying model sizes.

# also simaltaneously collect data for mean and std of energy and magnetization for both methods and plot 
# relative error as func of N

beta = 0.2
N_vals = [16, 32, 64, 128, 200, 256, 300, 350, 400, 450, 512] # N-values for which we want to compare time
J = 1.
H = 1.
sweep = true # see if meaningnful.
num_mc_steps = 1000 
num_model_steps = round(Int, ((test_size^2) / (model_size^2)) * num_mc_steps) 

model_path = "beta=$(beta)_model.jld2"
model, _ = NeuralIsing.initialize_default_model()
NeuralIsing.load_model!(model, model_path)

mcmc_times = Float64[]
mcmc_taus = Float64[]
mcmc_energy_means = Float64[]
mcmc_mag_menas = Float64[]

model_times = Float64[]
model_taus = Float64[]
model_energy_means = Float64[]
model_mag_means = Float64[]


t1, t2, t3, t4 = time(), time(), time(), time() # initialize for efficiency

for N in N_vals
    println("Running for N=$(N)")

    # initialize system and hamiltonian
    local system, hamiltonian = NeuralIsing.initialize_sys(N, J, H)

    # time MCMC run
    local t1 = time()
    local mcmc_systems, mcmc_magnetization, mcmc_energies = NeuralIsing.run_mc(system, hamiltonian, 
            NeuralIsing.SingleSpinFlip(), beta, num_mc_steps, NeuralIsing.compute_magnetization, sweep)
    local t2 = time()
    push!(mcmc_times, t2 - t1)

    local tau_mcmc, mcmc_ind_sys, mcmc_ind_mag = NeuralIsing.analyze_mc_chain(mcmc_systems, mcmc_magnetization)
    push!(mcmc_taus, tau_mcmc)
    push!(mcmc_energy_means, mean(mcmc_energies))
    push!(mcmc_mag_menas, mean(mcmc_ind_mag))
    # println("MCMC time: $(t2 - t1) seconds")

    # time model based run
    local t3 = time()
    local model_energies, model_magnetization, model_systems, update_indices, probabilities = NeuralIsing.test_model(model, 16, N, num_model_steps, "random", beta)
    local t4 = time()

    push!(model_times, t4 - t3)
    local tau_model, model_ind_sys, model_ind_mag = NeuralIsing.analyze_mc_chain(model_systems, model_magnetization)
    push!(model_taus, tau_model)
    push!(model_energy_means, mean(model_energies))
    push!(model_mag_means, mean(model_ind_mag))
end

# compute time-per-indpenednet sample for each method (wall-clock time / num independent samples = time / (num_steps / tau) = time * tau / num_steps)
mcmc_time_per_ind = mcmc_times .* mcmc_taus ./ num_steps
model_time_per_ind = model_times .* model_taus ./ num_steps

plt = plot(xlabel="Lattice Size N", ylabel="Time per Independent Sample (s)", title="Time Complexity Comparison: MCMC vs Model", legend=:topleft)
plot!(plt, N_vals, mcmc_time_per_ind, yscale = :log10, label="MCMC", lw=2, marker=:circle)
plot!(plt, N_vals, model_time_per_ind,  yscale = :log10, label="Model Based", lw=2, marker=:star)

filename = "experiments/fig/complexity_comparison_beta=$(beta).png"
savefig(plt, filename)

# relative errors in magnetization and energy
error_mag = abs.(mcmc_mag_menas .- model_mag_means) 
rel_error_energy = abs.(mcmc_energy_means .- model_energy_means) ./ abs.(mcmc_energy_means)

plt2 = plot(layout=(2,1), size=(800,800), title="Errors in Mean Energy: MCMC vs Model", legend=:topright)

plot!(plt2[1], N_vals, error_mag,  lw=2, marker=:circle, xlabel="Lattice Size N", ylabel="Absolute Error", title="Magnetization")
plot!(plt2[2], N_vals, rel_error_energy, lw=2, marker=:star, xlabel="Lattice Size N", ylabel="Relative Error", title="Energy")

filename2 = "experiments/fig/relative_errors_beta=$(beta).png"
savefig(plt2, filename2)




