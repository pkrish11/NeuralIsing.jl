# ML functionalitiy
using DelimitedFiles
using Flux
using LinearAlgebra
using Random
using JLD2


"""
    get_training_data(num_runs::Int, data_type::String, beta::Float64,
                      batch_size::Int, k::Int)

Collect and assemble training batches from multiple Monte Carlo runs at a fixed β.

This function loads stored spin configurations for each run, constructs
input–target pairs suitable for neural network training, and aggregates
them into shuffled mini-batches. 

This function is top-level, calls to fetch_data_for_run for actually structuring the data and computing targets.

# Arguments
- `num_runs::Int`: Number of independent Monte Carlo runs to load.
- `data_type::String`: Type of stored data (e.g. `"raw"` or `"independent"`). Must be these strings exactly if expecting function to read from disk.
- `beta::Float64`: Inverse temperature at which the data was generated.
- `batch_size::Int`: Number of samples per training batch.
- `k::Int`: Group size used to construct targets from correlated samples.

# Returns
- `training_data`: Vector of batches of the form `(x, β, target)`, where
  `x` is the input configuration tensor and `target` is the corresponding
  target probability tensor.

# Notes
Batches are shuffled before being returned to remove correlations between runs and temperatures.
"""
function get_training_data(num_runs::Int, data_type::String, beta::Float64, batch_size::Int, k::Int)
   
    all_data = [] # get data from each run 

    for run in 1:num_runs
        data_beta = fetch_data_for_run(run, data_type, beta, batch_size, k)
        for data in data_beta
            push!(all_data, data)
        end
    end
    shuffle!(all_data) # shuffle batches.

    return all_data
end


"""
    fetch_data_for_run(run, data_type, beta, batch_size::Int, k)

Load and structure training data from a single Monte Carlo run. 
Lower-level function, called to by get_training_data. 

Reads stored spin configurations from disk, reshapes them into 4D tensors,
calculates target for NN, and returns appropriately structured batch data.

This function constructs the target of the NN as (Sum i=1 to k of boolean(y_i - y_0)) / k 
to approximate the site-wise flip probability between model states.

# Arguments
- `run`: Index of the Monte Carlo run.
- `data_type`: Type of stored data (e.g. `"raw"` or `"independent"`).
- `beta`: Inverse temperature associated with the data.
- `batch_size::Int`: Number of samples per batch.
- `k::Int`: Number of configurations grouped together to compute targets.

# Returns
- `batches`: Vector of training batches. Each batch is a tuple
  `(x_batch, β_batch, target_batch)` suitable for Flux training.

# Notes
- Input tensors have shape `(L, L, 1, batch_size)`, suitable for CNN learning. User should change number-of-chanels as they see fit.
-User should change the target_computation as they see fit.
"""
function fetch_data_for_run(run, data_type, beta, batch_size::Int, k)
   
    filename = "data/$(data_type)/$(beta)/samples_run=$(run).txt"
    data = readdlm(filename, ',')
    N, l_squared = size(data)
    l = convert(Int, sqrt(l_squared))

    # modify data to have shape (l,l,1,N)
    data = reshape(data, l, l, 1, N)

    # divide into groups based on k-size.
    num_groups = floor(Int, N // k)
    grouped_data = [Float32.(data[: , : , : , (b-1)*k+1 : b*k])
                 for b in 1:num_groups] 

    # compute target and add beta/target for each group, and pick site that is the "first one" randomly for each group
    new_data_vec = []
    for group in grouped_data
        site = 1 # could also chose other.
        x = reshape(group[:,:,:,site], l,l,1,1) # explicitly enforce 4d input with 1 chanel and 1 batch-size.
        target_prob = sum(abs.(group[:,:,:,setdiff(1:k, site)] .- x)./2, dims=4) / (k-1)
        push!(new_data_vec, (x, Float32(beta) , Float32.(target_prob)))
    end

    num_batches = floor(Int, length(new_data_vec) / batch_size)
    batches = []
    for i in 1:num_batches
        batch = new_data_vec[(i-1)*batch_size + 1 : i*batch_size]
        
        # stack inputs, betas, targets along the last dimension (batch dimension)
        x_batch = cat([b[1] for b in batch]..., dims=4)         # shape: l x l x 1 x batch_size
        β_batch = Float32.([b[2] for b in batch])             # shape: batch_size
        t_batch = cat([b[3] for b in batch]..., dims=4)       # shape: l x l x 1 x batch_size (or whatever target shape)
    
        push!(batches, (x_batch, reshape(β_batch, 1, batch_size), t_batch))
    end

    return batches
end



"""
    initialize_default_model()

Initialize the default convolutional encoder–decoder model and optimizer. 
Encoder 

Constructs a CNN-based encoder that maps spin configurations to a latent
representation and a decoder that reconstructs per-site flip probabilities
conditioned on inverse temperature β.

# Returns
- `model`: A callable model `model(x, β)` producing site-wise flip probabilities.
- `opt`: ADAM optimizer initialized for the model parameters.

# Notes
- The latent dimension is 64.
- The inverse temperature β is concatenated to the latent vector before decoding.
Model has the structure : 
Encoder = Chain(
  Conv((3, 3), 1 => 16, relu, pad=1),   # 160 parameters
  MaxPool((2, 2)),
  Conv((3, 3), 16 => 32, relu, pad=1),  # 4_640 parameters
  MaxPool((2, 2)),
  FinalProject.var"#58#60"(),
  Dense(512 => 64, relu),               # 32_832 parameters
)                   # Total: 6 arrays, 37_632 parameters, 147.602 KiB.
Chain(
  Dense(65 => 512, relu),               # 33_792 parameters
  FinalProject.var"#59#61"(),
  ConvTranspose((2, 2), 32 => 16, relu, stride=2),  # 2_064 parameters
  ConvTranspose((2, 2), 16 => 1, σ, stride=2),  # 65 parameters
)                   # Total: 6 arrays, 35_921 parameters, 140.824 KiB.
"""
function initialize_default_model()
    encoder = Chain(
    # → (16,16,1,N)
    Conv((3,3), 1=>16, relu; pad=1),
    MaxPool((2,2)),             # → (8,8,16,N)

    Conv((3,3), 16=>32, relu; pad=1),
    MaxPool((2,2)),             # → (4,4,32,N)

    x -> reshape(x, (4*4*32, size(x, 4))),   # → (512, N)

    Dense(512, 64, relu)        # latent dimension
    )

    decoder = Chain(
    Dense(65, 512, relu),  # 65th input here is the temp param.  z(64) + β(1)
    x -> reshape(x, 4, 4, 32, :),  # change last to : to get automatic resizing ; restore feature map

    # 4x4 → 8x8
    ConvTranspose((2,2), 32 => 16, relu; stride=2, pad=0),

    # 8x8 → 16x16
    ConvTranspose((2,2), 16 => 1, sigmoid; stride=2, pad=0)
    )

    model(x,β) = decoder(vcat(encoder(x), β))

    
    opt = Flux.setup(Adam(), model)

    return model, opt
end

"""
    train_model!(model, opt, loss_fn, training_data, epochs)

Train a neural network model using Flux. Runs gradient-based optimization over the provided training data for a
specified number of epochs and reports elapsed training time. 

User may choose their own model, optimizer and loss function, as long as they are all compatible with each other and training data.

# Arguments
- `model`: Flux model to be trained.
- `opt`: Optimizer specialized on the `model`'s parameters.
- `loss_fn`: Loss function, should be compatible with training_data strucutre and model outputs.
- `training_data`: Iterable of training batches.
- `epochs::Int`: Number of training epochs.

# Returns
- `nothing`

# Notes
This function mutates `model` in place.
"""
function train_model!(model, opt, loss_fn, training_data, epochs)
    
    println("starting training")
    t1 = time()
    for epoch in 1:epochs
        Flux.train!(loss_fn, model, training_data, opt)
        yield()
        print(epoch)
    end
    t2 = time()

    println("\ndone training")
    println("Time taken: $(t2 - t1) seconds")

end

"""
    prob_loss(model, x, β, y)

Default loss function, using MSE between target flip probability and predicted (model output), 
plus expected magnetization vs predicted.

# Arguments
- `model`: Trained or training model.
- `x`: Input spin configuration tensor.
- `β`: Inverse temperature associated with the input.
- `y`: Target flip-probability tensor.

# Returns
- `loss::Real`: Scalar loss value.
"""
function prob_loss(model, x, β, y)

    model_probs = model(x, β)
    target_probs = y
    prob_loss = Flux.mse(target_probs, model_probs) 
    λ_prob = 1
    
    N= size(x)[1] * size(x)[2]
    pred_mag = sum(x .* (1 .- 2 .*model_probs)) / N
    expected_mag = sum(x .* (1 .- 2 .*target_probs)) / N
    mag_loss = abs(expected_mag-pred_mag)
    λ_mag =  0.1

    return (λ_prob*prob_loss + λ_mag*mag_loss)

end


"""
    save_model(name::String, model)

Save the encoder and decoder parameters of a trained model to disk, in a .jld2 file, under /model subdirector.

# Arguments
- `name::String`: Filename used to store the model parameters.
- `model`: Trained model containing `encoder` and `decoder`.

# Returns
- `nothing`

# Notes
Model parameters are saved using JLD2 and can be restored with `load_model!`.
"""
function save_model(name::String, model)
    if !isdir("model")
        mkdir("model")
    end
    filename = "model/"*name
    jldsave(filename; encoder=Flux.state(model.encoder), decoder=Flux.state(model.decoder))
end

"""
    load_model!(model, name::String)

Load saved encoder and decoder parameters into an existing model.

# Arguments
- `model`: Initialized model whose parameters will be overwritten.
- `name::String`: Filename from which parameters are loaded.

# Returns
- `nothing`

# Notes
The architecture of `model` must match the saved parameters.
"""
function load_model!(model, name::String)
    filename = "model/"*name
    loaded = JLD2.load(filename)
    Flux.loadmodel!(model.encoder, loaded["encoder"])
    Flux.loadmodel!(model.decoder, loaded["decoder"])
end


"""
    function magnetization(spins)
Compute magnetization for the given spins.
"""
function magnetization(spins)
    return sum(spins) / length(spins)
end

"""
    function get_training_data(num_runs::Int, data_type::String, beta_vals::Vector{Float64}, batch_size::Int, k::Int)

Specialization of the get_training_data() function to the case when several beta_values are being trained for. Reads data for all beta_values
    and returns appropriately structured (input output) pairs, calling the fetch_data_for_run function for each beta,run.
"""
function get_training_data(num_runs::Int, data_type::String, beta_vals::Vector{Float64}, batch_size::Int, k::Int)
   
    all_data = []

    for beta in beta_vals
        for run in num_runs
            data_beta = fetch_data_for_run(run, data_type, beta, batch_size, k)
            for data in data_beta
                push!(all_data, data)
            end
        end
    end

    # shuffle batches.
    shuffle!(all_data)

    return all_data
end