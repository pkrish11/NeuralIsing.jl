using NeuralIsing
using Plots
Random.seed!(1234)

plot_betas = true
plot_energy_posterior = true
plot_magnetization_posterior = true
beta_to_plot_for = 0.2

# define a set of beta-values across which we want to run MCMC, get energy and magnetization evolutions, and 
# plot the mean and standard-dev energy and magnetization as functions of beta.

beta_vals = [0.01*i + 0.05 for i in 0:79] # beta uniformly spaced from 0.05 to 0.84
move = NeuralIsing.SingleSpinFlip()
num_steps = 3000
obs = NeuralIsing.compute_magnetization
sweep=true
model_size, J, H = 16, 1., 1.

std_energies = Float64[]
std_mag = Float64[]
mean_energy = Float64[]
mean_mag = Float64[]
autocor_time = Float64[]

beta_plot_states, beta_plot_mag, beta_plot_energy = 0, 0, 0
if !isdir("experiments/fig")
    mkdir("experiments/fig")
end

# for each beta val, run MCMC for several steps, extract independent samples and store 
# mean and standard deviations in energy and magnetization and autocore time. 
for beta in beta_vals
    system, hamiltonian = NeuralIsing.initialize_sys(model_size, J, H)
    system_evol, mag_evol, energy_evol = NeuralIsing.run_mc(system, hamiltonian, move, beta, num_steps, obs, sweep)

    tau, sys, ind_mag, ind_energy = NeuralIsing.analyze_mc_chain(system_evol, mag_evol, energy_evol)

    push!(std_energies, std(ind_energy))
    push!(std_mag, std(ind_mag))
    push!(mean_energy, mean(ind_energy))
    push!(mean_mag, mean(ind_mag))
    push!(autocor_time, tau)

    # for the specific beta we want to plot posterior distributions for, 
    # make the energy and magnetization posterior plots.
    if beta == beta_to_plot_for
        if plot_energy_posterior
            savefig(histogram(ind_energy, bins=30, normalize=true, title="Energy Posterior at beta=$(beta_to_plot_for)", 
                    xlabel="Energy", ylabel="Density"), 
                    "experiments/fig/energy_posterior_beta=$(beta_to_plot_for).png")
        end
        if plot_magnetization_posterior
            savefig(scatter(ind_mag, title="Magnetization Evolution at beta=$(beta_to_plot_for)"
                    ,xlabel="Samples", ylabel="Magnetization"),
                    "experiments/fig/magnetization_posterior_beta=$(beta_to_plot_for).png")
        end
    end
end

# plot these 5 variables as function of beta_vals
if plot_betas
    plt = plot(layout = (5, 1), size=(600, 1200))
    # 1 
    scatter!(plt[1], beta_vals, std_mag, markersize=5, label="Obs Std")
    ylabel!(plt[1], "Obs Std")
    # 2
    scatter!(plt[2], beta_vals, mean_mag, markersize=5, label="Obs Mean")
    ylabel!(plt[2], "Obs Mean")
    # 3
    scatter!(plt[3], beta_vals, std_energies, markersize=5, label="Energy Std")
    ylabel!(plt[3], "Energy Std")
    # 4
    scatter!(plt[4], beta_vals, mean_energy, markersize=5, label="Energy Mean")
    ylabel!(plt[4], "Energy Mean")
    # 5
    scatter!(plt[5], beta_vals, autocor_time, markersize=5, label="Autocorrelation time")
    ylabel!(plt[5], "Autocorrelation time")

    if !isdir("experiments/fig")
        mkdir("experiments/fig")
    end
    savefig(plt, "experiments/fig/swoop_betas.png")
end

