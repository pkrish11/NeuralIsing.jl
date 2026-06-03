using Random

abstract type AbstractMCMove end

"""
    struct SingleSpinFlip <: AbstractMCMove

Struct holding information about the type of MC Move to be proposed.
Single Spin Flip ==> picking a random spin and flipping according to Metropolois acceptance criterion.
"""
struct SingleSpinFlip <: AbstractMCMove end

struct SpinFlipInfo
    index::Int
end

abstract type AbstractSystem end

abstract type AbstractLattice end

abstract type AbstractEnergy end

# define 2d lattice structure, constructor, neighbour finding function
struct twoDLattice <: AbstractLattice
    dims::NTuple{2,Int}         # size of  2D lattice in each dimension (must be a rectangular system)
    neighbors::Vector{Int} # scratch space to store neighbours of current chosen index
    neighbor_list::Vector{Vector{Int}} # list of neighbours of each lattic site (as indices)
end


""" 
    function twoDLattice(size::Int)

Constructor function for the twoDLattice struct.

# Arguments
- `size::Int`: Size of the two-dimensional lattice to be constructed.

# Returns
- `twoDLattice`: twoDLattice struct of dimensions size x size, with neighbour_list precomputed.
"""
function twoDLattice(size::Int)
    dims = (size,size)
    D = 2 # since 2 d lattice
    N = prod(dims) # get total number of lattice sites
    neighs = Vector{Int}(undef,2*D) # define scratch vector
    neighbor_list = [Vector{Int}() for _ in 1:N] # initialize list of neighbours
    cartesian_indices = CartesianIndices(dims) # get cartesian and linear indices of the lattice
    linear_indices = LinearIndices(dims)
    for i in 1:N
        current_idx = cartesian_indices[i] # for each lattice site, get the current index
        for d in 1:D                       # and along each dimension, compute -1 and +1 offsets as neighbours and push to list.
        for offset in (-1, 1)
            mod_offset = mod1(current_idx[d] + offset,dims[d])
            neigh_idx = Base.setindex(current_idx, mod_offset, d)
            push!(neighbor_list[i], linear_indices[neigh_idx])
        end
        end
    end
    twoDLattice(dims, neighs, neighbor_list)
end

""" 
    function neighbors!(lattice::twoDLattice, site::Int) 

Function to assign the correct vector of neighbours, from the "neighbour_list", to the scratch "neighbours" space.

# Arguments
- `size::Int`: Size of the two-dimensional lattice to be constructed.

# Returns
- `twoDLattice`: twoDLattice struct of dimensions size x size, with neighbour_list precomputed.
"""
function neighbors!(lattice::twoDLattice, site::Int) 
    lattice.neighbors .= lattice.neighbor_list[site]
end

""" 
    struct SpinSystem{T, L <: AbstractLattice} <: AbstractSystem
    
Spin System Struct that holds the 1-dimensional vector of spins, and the lattice object of type Abstract Lattice.

# Attributes    
spins::Vector{T}
lattice::L

"""
struct SpinSystem{T, L <: AbstractLattice} <: AbstractSystem
    spins::Vector{T}
    lattice::L
end

"""
    function _random_spin_state(::Type{T}, N) where {T <: Integer}

Helper function to get a random spin state.
"""
function _random_spin_state(::Type{T}, N) where {T <: Integer}
    return T.(rand(Bool, N) .* 2 .- 1)
end

"""
    function SpinSystem(lattice::L; initial_state=:random) where {L<:AbstractLattice}

Constructor function for Spin system, with random or alligned (+1) states. 

Takes a 2D lattice as input and returns SpinSystem object as output.
"""
function SpinSystem(lattice::L; initial_state=:random) where {L<:AbstractLattice}
    N = num_sites(lattice)
    if initial_state == :random
        spins = _random_spin_state(Int,N) 
    else 
        spins = ones(Int, N)
    end
    return SpinSystem(spins, lattice)
end

# function to get num_sites from lattice object
function num_sites(lattice::AbstractLattice)
    return prod(lattice.dims)
end


# Propose a random spin to flip --> propose move method specialized on Spin System and Single Spin Flip Move
function propose_move(system::SpinSystem{<:Integer}, hamiltonian::AbstractEnergy,
    move::SingleSpinFlip)
random_index = rand(1:length(system.spins))
return SpinFlipInfo(random_index)
end

# function to accept move and flip spin at proposed index
function accept_move!(system::SpinSystem{T}, move_info::SpinFlipInfo, move::SingleSpinFlip) where {T <: Integer}
system.spins[move_info.index] *= -one(T)
end

function metropolis_step!(system, hamiltonian, move::AbstractMCMove, beta::Float64)
    trial_move_info = propose_move(system, hamiltonian, move)
    
    ΔE = calculate_ΔE(system, trial_move_info, hamiltonian, move)
    
    if ΔE <= 0.0 || rand() < exp(-beta * ΔE)
        accept_move!(system, trial_move_info, move)
        return true
    end
    return false
end


function mc_sweep!(system::SpinSystem, hamiltonian, move, beta)
    for _ in eachindex(system.spins)
        metropolis_step!(system,hamiltonian,move, beta)
    end
end

# function to compute magnetization of the system
function compute_magnetization(sys::SpinSystem{T}) where {T <: Integer}
    N = length(sys.spins)
    val = 0.0
    for s in sys.spins
        val += s / N
    end
    return val
end

"""
    struct IsingHamiltonian <: AbstractEnergy
Struct to represent the Ising Hamiltonian , holds the J (interaction energy term) and H (field energy term) values.
"""
struct IsingHamiltonian <: AbstractEnergy
    J::Float64
    h::Float64
end

"""
    function calculate_ΔE(system::SpinSystem, move_info::SpinFlipInfo, ham::IsingHamiltonian, move::SingleSpinFlip)

Function to calculate change in energy of the spin system, given a proposed single spin flip.
"""
function calculate_ΔE(system::SpinSystem, move_info::SpinFlipInfo, 
                       ham::IsingHamiltonian, move::SingleSpinFlip)
    site_idx = move_info.index
    neighbors!(system.lattice, site_idx)
    current_spin = system.spins[site_idx]

    neighbor_spin_sum = sum(@view system.spins[system.lattice.neighbors])
    
    ΔE = 2.0 * ham.J * current_spin * neighbor_spin_sum + 2.0 * ham.h * current_spin
    return ΔE
end

""" 
    function total_energy(system::SpinSystem, hamiltonian::IsingHamiltonian)

Function to calculate the total energy of a spin system, using a brute force approach.

"""
function total_energy(system::SpinSystem, hamiltonian::IsingHamiltonian)
    interaction_energy = 0.0

    for i in 1:num_sites(system.lattice)
        s_i = system.spins[i]

        neighbors!(system.lattice, i)

        for j in system.lattice.neighbors
            if j > i
                s_j = system.spins[j]
                interaction_energy -= hamiltonian.J * s_i * s_j
            end
        end
    end

    field_energy = -hamiltonian.h * sum(system.spins)

    return interaction_energy + field_energy
end


