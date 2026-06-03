# example of training a single model on data from several different temperatures.

using NeuralIsing
using Flux

beta_vals = [0.1*i + 0.1 for i in 0:7] # pick betas from 0.1 to 0.8 which model is trained on.
J = 1.
H = 1.
model_size = 16
system_params = (model_size, J, H)

num_steps = 3000
observable = NeuralIsing.compute_magnetization
sweep = true
num_runs = 10 # number of times to run MCMC independently, with different initial configurations.
mc_details = (num_steps, observable, sweep)
move = NeuralIsing.SingleSpinFlip()

# generate data for training, independent samples from MCMC, for num_runs different initial configurations.
# for each beta value.
for beta in beta_vals
    NeuralIsing.generate_raw_data(system_params, mc_details, num_runs, beta, move; save_raw=true)
end

# fetch training data, shuffled and set into batches
data_type = "independent" 
batch_size = 40
k = 5 # number of samples to group together for target computation (feature of how we structure the loss function.)

# can also be run with several beta values, in which case model learns to generalize over beta.
train_data = NeuralIsing.get_training_data(num_runs, data_type, beta_vals, batch_size, k)

# define model. Can be any Flux.jl object that takes in 
# (x,beta) as input and returns y=(prob of flip sitewise.)
# the default model is a simple CNN with beta concatenated at the dense layer.
# the default optimizer is ADAM, with default parameters.
model, opt = NeuralIsing.initialize_default_model()

# the loss function, in our architecture, is MSE between predicted flip probabilities and 
# target probabilities (computed using k-samples grouped together from MCMC data.)
# One could use any other loss function as well, keeping in mind to also modify the training data accordingly.
loss = NeuralIsing.prob_loss 

# run the training function, for specificed number of epochs.
num_epochs = 50
NeuralIsing.train_model!(model, opt, loss, train_data, num_epochs)

# save the model in a JLD2 file, in "/model" directory.
name = "manyBetaIndData_Model.jld2"
NeuralIsing.save_model(name, model)