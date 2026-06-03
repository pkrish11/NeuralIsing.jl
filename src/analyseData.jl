using DelimitedFiles
using StatsBase

"""
    generate_raw_data(system_params, mc_details, num_runs, beta, move; save_raw=false)

Run multiple MC simulations (using Single Spin Flips) and collect raw states and observable data.
Also extract independent states and observable data, and save to disk the independent data and samples (always) and raw data(if requested).

# Arguments
- `system_params`: Parameters defining the physical system, Tuple of (system size, J, H) where J,H define Ising Hamiltonian
- `mc_details`: Monte Carlo configuration parameters : (number of MC steps, Observable Function, Boolean of sweep))
- `num_runs::Int`: Number of independent Monte Carlo runs to perform.
- `beta::Real`: Inverse temperature β = 1 / T.
- `move`: Monte Carlo update rule (e.g. single-spin flip).
- `save_raw::Bool=false`: If `true`, also save raw system configurations and observables for each run to disk.

# Returns
- `samples`: Collection of sampled system configurations.
- `observables`: Collection of measured observables for each run, computed from the inputted Observable function.

# Notes
Each run is initialized independently, so returned samples should be statistically independent up to equilibration and autocorrelation effects of the chosen move.
"""

function generate_raw_data(system_params, mc_details, num_runs, beta, move; save_raw=false)
    
    num_steps, observable, sweep = mc_details  
    size, J, H = system_params

    for run in 1:num_runs
        system, hamiltonian = initialize_sys(size, J, H)                # initialize a new system for each run to get more randomness in model. 
        system_init = copy(system)                                      # store a copy of the system 
        system_evolution, observable_evolution, time_to_run = run_mc(system_init, hamiltonian, move, beta, num_steps, observable, sweep)
        tau_time, ind_samples, ind_obs = analyze_mc_chain(system_evolution, observable_evolution)            # get independent samples from data.
        store_data(ind_samples, ind_obs, run, beta, raw=false)                                               # store independent data in CSV file.
        
        if save_raw
            store_data(system_evolution, observable_evolution, run, beta, raw=true) 
        end
    end
    return true 
end

"""
    store_data(samples, obs, run, beta; raw=false)

Helper function (internal) to store MC samples and observables to disk, for independent and raw.
Sets up data directories also, and file-naming conventions.

# Notes
This function performs I/O only and does not modify the input data.
"""
function store_data(samples, obs, run, beta; raw=false)
    if !isdir("data")
        mkdir("data")
    end

    if !isdir("data/independent")
        mkdir("data/independent")
    end
    if !isdir("data/raw")
        mkdir("data/raw")
    end
    if !isdir("data/independent/$(beta)")
        mkdir("data/independent/$(beta)")
    end
    if !isdir("data/raw/$(beta)")
        mkdir("data/raw/$(beta)")
    end

    data_type = "independent" 
    if (raw) data_type = "raw" end

    file_name_states = "data/$(data_type)/$(beta)/samples_run=$(run).txt"
    open(file_name_states, "w") do io
        mat = transpose(hcat(samples...)) # convert vector of matrices to single matrix for writedlm
        writedlm(io, mat, ',')
    end

    file_name_observable = "data/$(data_type)/$(beta)/magnetization_run=$(run).txt"
    open(file_name_observable, "w") do io
        writedlm(io, obs, ',')
    end
end

"""
    initialize_sys(size::Int, J::Float64, H::Float64)

Helper function to initialize a 2D spin-system, given the lattice-size (square), J and H parameters defining the Hamiltonia.
Returns a System and Hamiltonian object.
"""
function initialize_sys(size::Int, J::Float64, H::Float64)
    lattice = twoDLattice(size)
    system = SpinSystem(lattice; initial_state=:random) # initialized with random states
    hamiltonian = IsingHamiltonian(J, H)

    return system, hamiltonian
end

"""
    run_mc(system, hamiltonian, move, beta, num_steps, observable, sweep)

Run a single Monte Carlo simulation for a given system and Hamiltonian.

Evolves the system using the specified Monte Carlo `move` at inverse temperature `beta`, recording observables and energies over the course
of the simulation.

# Arguments
- `system`: Initial system configuration.
- `hamiltonian`: Hamiltonian defining the system energy.
- `move`: Monte Carlo update rule.
- `beta::Real`: Inverse temperature β = 1 / T.
- `num_steps::Int`: Total number of Monte Carlo steps.
- `observable`: Observable(s) to measure during the simulation.
- `sweep::Int`: Number of updates per Monte Carlo sweep.

# Returns
- `system_evolution`: Time series of system configurations.
- `observable_evolution`: Time series of measured observables.
- `energy_evolution`: Time series of system energies.

# Notes
For local updates, one sweep corresponds to `N` attempted updates, where `N` is the number of lattice sites.
"""
function run_mc(system, hamiltonian, move, beta, num_steps, observable, sweep)

    all_system = []             # vector to store data over MC simulation.l
    all_observable = Float64[]  
    all_energy = Float64[]

    for i in 1:num_steps
        push!(all_system, copy(system.spins)) # push system and observable from previous step to vectors
        push!(all_observable, observable(system))
        push!(all_energy, total_energy(system, hamiltonian))

        # run MC to get new system
        if sweep 
            mc_sweep!(system, hamiltonian, move, beta)
        else 
            metropolis_step!(system, hamiltonian, move, beta)
        end
    end

    return all_system, all_observable, all_energy
end

""" Specialize copy function for system, to return a copy of the spins. """
Base.copy(s::SpinSystem) = SpinSystem(copy(s.spins), s.lattice) 

"""
    analyze_mc_chain(system_evolution, observable_evolution, energy_evolution)

Analyze Monte Carlo time series data by computing autocorrelation times in the observable.
Extract and return corresponding `indpendent samples` for system, observable, and energy vectors. 
`Independent Samples` are those seperated by atleast 2*autocorelation time

Autocorrelation time is computed as :
- Compute autocorelation between subsequent observable values, using Julia Stats Library. 
- Use the basic algorithm described by [1] to compute the autocorelation time.

# Arguments
- `system_evolution`: Time series of system configurations, as a list of vectors.
- `observable_evolution`: Time series of measured observables, as Vector{Float64}
- `energy_evolution`: Time series of system energies, as Vector{Float64}

# Returns
- S : computed autocorrelation time
- independent_system_samples : Samples from `system_evolution` vec seperated by atleast 2*S 
- independent_observables : Observable values corresponding to the `independent` system states, as Vector{Float64}
- independent_energy : Energy values corresponding to the `independent` system states, as Vector{Float64}

References : 
[1] : Madras, N. & Sokal, A. D. (1988). The Pivot Algorithm: A Highly Efficient Monte Carlo Method for the Self-Avoiding Walk. Journal of Statistical Physics, 50(1–2), 109–186. https://doi.org/10.1007/BF01022990
"""
function analyze_mc_chain(system_evolution, observable_evolution, energy_evolution)

    # get ACF functions using StatsBase
    n = length(observable_evolution)
    max_autocors = min(n, Int(floor(10*log10(n)))) # based on Julia's suggested default range.
    range_acfs = collect(1:max_autocors)

    float_obs = Float64.(observable_evolution) # ensure observable is Float64 type for StatsBase

    autocors = StatsBase.autocor(float_obs, range_acfs; demean=true) 

    if any(isnan.(autocors))
        S = 0.0
        return S, system_evolution, observable_evolution, energy_evolution
    else 
        S = 1 # running sum
        for w in 1:length(autocors)
            S += 2*autocors[w] 
            if (w >= 6*S) # stopping criteria
                break
            end
        end
        gap = ceil(Int, 2*S) # use rounded up 2*τ_int as indexing gap

        return S, system_evolution[1:gap:end, :], observable_evolution[1:gap:end], energy_evolution[1:gap:end]
    end
end


"""
    analyze_mc_chain(system_evolution, observable_evolution, energy_evolution)

Analyze Monte Carlo time series data by computing autocorrelation times in the observable.
Extract and return corresponding `indpendent samples` for system and observables.
`Independent Samples` are those seperated by atleast 2*autocorelation time

Autocorrelation time is computed as :
- Compute autocorelation between subsequent observable values, using Julia Stats Library. 
- Use the basic algorithm described by [1] to compute the autocorelation time.

# Arguments
- `system_evolution`: Time series of system configurations, as a list of vectors.
- `observable_evolution`: Time series of measured observables, as Vector{Float64}

# Returns
- S : computed autocorrelation time
- independent_system_samples : Samples from `system_evolution` vec seperated by atleast 2*S 
- independent_observables : Observable values corresponding to the `independent` system states, as Vector{Float64}

References : 
[1] : Madras, N. & Sokal, A. D. (1988). The Pivot Algorithm: A Highly Efficient Monte Carlo Method for the Self-Avoiding Walk. Journal of Statistical Physics, 50(1–2), 109–186. https://doi.org/10.1007/BF01022990
"""
function analyze_mc_chain(system_evolution, observable_evolution)

    # get ACF functions using StatsBase
    n = length(observable_evolution)
    max_autocors = min(n, Int(floor(10*log10(n)))) # based on Julia's suggested default range, find specific reference.
    range_acfs = collect(1:max_autocors)

    autocors = StatsBase.autocor(observable_evolution, range_acfs; demean=true) 

    if any(isnan.(autocors))
        S = 0.0
        return S, system_evolution, observable_evolution
    else 

        S = 1 # running sum
        for w in 1:length(autocors)
            S += 2*autocors[w] 
            if (w >= 6*S) # stopping criteria
                break
            end
        end
        gap = ceil(Int, 2*S)     #use rounded up 2*τ_int as indexing gap

        return S, system_evolution[1:gap:end, :], observable_evolution[1:gap:end]
    end
end




