using NeuralIsing
using Plots

beta = 1.1
num_mc_steps = 5000 
J = 1.
H = 1.

model_size = 16
test_size = 16       # function that allows model implementation on larger lattices also.
model_path = "beta=$(beta)_model.jld2"
num_model_steps = round(Int, ((test_size^2) / (model_size^2)) * num_mc_steps) 
 # scale number of steps to keep same number of sweeps

# load model
model, _ = NeuralIsing.initialize_default_model()
NeuralIsing.load_model!(model, model_path)

# run regular MCMC and get magnetization and energy distributions. 
system, hamiltonian = NeuralIsing.initialize_sys(test_size, J, H)
mcmc_systems, mcmc_magnetization, mcmc_energies = NeuralIsing.run_mc(system, hamiltonian, 
        NeuralIsing.SingleSpinFlip(), beta, num_mc_steps, NeuralIsing.compute_magnetization, true)

mcmc_tau, mcmc_ind_sys, mcmc_ind_mag, mcmc_ind_energy = NeuralIsing.analyze_mc_chain(mcmc_systems, mcmc_magnetization, mcmc_energies)
println("Regular MCMC autocorrelation time: $(mcmc_tau)")
println("Regular MCMC mean and standard deviation magnetization: $(mean(mcmc_ind_mag)), $(std(mcmc_ind_mag))")
println("Regular MCMC mean and standard deviation energy: $(mean(mcmc_ind_energy)), $(std(mcmc_ind_energy))")

# run model based simulation for same number of steps, get magnetization and energy distributions.
# note that this function will also initialize the system randomly, like the run_mcmc function.

model_energies, model_magnetization, model_systems, update_indices, acceptance_prob = NeuralIsing.test_model(model, model_size, test_size, num_model_steps, "random", beta, J, H)

model_tau, model_ind_sys, model_ind_mag, model_ind_energy = NeuralIsing.analyze_mc_chain(model_systems, model_magnetization, model_energies)
# model_ind_sys, model_ind_mag, model_ind_energy, model_tau = model_systems, model_magnetization, model_energies, 1
println("Model based autocorrelation time: $(model_tau)")
println("Model based mean and standard deviation magnetization: $(mean(model_ind_mag)), $(std(model_ind_mag))")
println("Model based mean and standard deviation energy: $(mean(model_ind_energy)), $(std(model_ind_energy))")
println("Model based acceptance probability: $(acceptance_prob)")

# make plot of enery and magnetization distributions, should look close for mcmc vs model.

mE, sE = mean(mcmc_ind_energy), std(mcmc_ind_energy)
modE, modEs = mean(model_ind_energy), std(model_ind_energy)

mM, sM = mean(mcmc_ind_mag), std(mcmc_ind_mag)
modM, modMs = mean(model_ind_mag), std(model_ind_mag)

# ------ Histogram of MCMC energies ------
p1 = histogram(
    mcmc_ind_energy,
    bins=40,
    alpha=0.4,
    label="MCMC",
    title="MCMC Energy Distribution",
)
# Mean energy lines
vline!(p1, [mE], label="MCMC mean", lw=2)
# ------ Histohram of Model energies ------
p2 = histogram(
    model_ind_energy,
    bins=40,
    alpha=0.4,
    label="Model",
    title="Model Energy Distribution",
)
# Mean energy lines
vline!(p2, [modE], label="Model mean", lw=2)
# ------ Scatter of magnetization ------
p3 = scatter(
    mcmc_ind_mag,
    alpha=0.6,
    xlabel="Samples ",
    title="MCMC Magnetization",
)
hline!(p3, [mM], label="MCMC mean", lw=2)
# ------ Scatter magnetization ------
p4 = scatter(
    model_ind_mag,
    alpha=0.6,
    xlabel="Samples ",
    title="Model Magnetization",
)
hline!(p4, [modM], label="Model mean", lw=2, ls=:dash)

# ------ 2×2 layout ------
plt = plot(p1, p2, p3, p4, layout=(2,2), size=(1000,800))

if test_size == model_size
    suptitle = "Validation on Same Size Lattice (Size=$(model_size), β=$(beta)).png"
else
    suptitle = "Validation on Larger Lattice (Model Size=$(model_size), Test Size=$(test_size), β=$(beta)).png"
end

savefig(plt,suptitle)




