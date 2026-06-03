# Expeiments on a model trained on MCMC data for 1 beta, but implemented on systems at different beta.

using NeuralIsing

beta = 0.2 # try value model is trained on
#test_betas = [0.1, 0.15, 0.19, 0.2, 0.21, 0.25, 0.3, 0.4] 

test_betas = [0.1*i+0.1 for i in 0:7] # beta values to test model on.
model_size = 16
num_steps = 1000
model_path = "manyBetaIndData_Model.jld2"

model, _ = NeuralIsing.initialize_default_model()
NeuralIsing.load_model!(model, model_path)

mag_means = Float64[]
energy_means = Float64[]
mags = []

test_size = 16

for test in test_betas
    local model_energies, model_magnetization, model_systems, update_indices, probabilities = NeuralIsing.test_model(model, model_size, test_size, num_steps, "random", test)
    local model_tau, model_ind_sys, model_ind_mag, model_ind_energy = NeuralIsing.analyze_mc_chain(model_systems, model_magnetization, model_energies)

    push!(mag_means, mean(model_ind_mag))
    push!(energy_means, mean(model_ind_energy))
    push!(mags, model_ind_mag)
end

plt = plot(layout = (length(test_betas),1), size=(500,1500), title="Model Generalization over Beta")
for i in eachindex(test_betas)
    scatter!(plt[i], mags[i], xlabel="Model Generated States", ylabel="Magnetization", title="Beta=$(test_betas[i])")
end

figname = "experiments/fig/model_generalization_crossBeta=multiModel_scatter.png"
savefig(plt, figname)

# fig of mean and std in magnetization and enery
plt2 = plot(layout = (2,1), size=(800,400), title="Mean Magnetization and Energy vs Beta")
scatter!(plt2[1], test_betas, mag_means, xlabel="Beta", ylabel="Mean Magnetization", title="Mean Magnetization vs Beta", label="Mean Magnetization", lw=2)
scatter!(plt2[2], test_betas, energy_means, xlabel="Beta", ylabel="Mean Energy", title="Mean Energy vs Beta", label="Mean Energy", lw=2)
figname2 = "experiments/fig/model_generalization_crossBeta=multiModel_FuncBeta.png"
savefig(plt2, figname2)

