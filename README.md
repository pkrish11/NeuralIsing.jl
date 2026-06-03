# NN based MCMC updates for 2D Ising Model

**FinalProject.jl** is a modular Julia framework for experimenting with Monte Carlo sampling of 2D lattice spin systems, using neural newtork based updates. 

 [ Please excuse my very poor choice of name for the package, unfortunately I did not realize this fatal mistake until I completely finished my project, and did not have enough time (before the deadline) to change this everywhere in the code :(   ]

---

## Motivation
Traditional "proposal-methods" for the 2D Ising model have known to become inefficient when the correlation lengths of the system are very large (near critical temperature) or when the systems themselves become very large. Producing independent samples becomes a slow task, especially for the cannonical "single-spin flip" based approach. 

This package proposes using neural network to instead update entire sub-lattices, ideally producing independent samples faster. Generalization of a single NN for systems across different temperatures, or Hamiltonian parameters, is also an appealing aspect of using NNs. 

---

## Features
The package mainly provides functionality that makes it easy to experiment with training neural networks for 2D Ising models. This includes:
* Efficiently generating training data : Running MCMC simulations (single spin flip updates) for any system or hamiltonian paramaters, and temperature. 
* Identifying and extracting independent samples from a MCMC simulation
* Integration with ML : computing target values, structuring data into batches for stochastic optimziation
* Default Loss Function and Model Architecture that are generalizable and would work-well for most systems
* Functions to implement NN based sub-updates on larger systems and generate MCMC simultations (while satisfying detailed balance)
* Routines for complexity and error analysis, and visualizing posterior distribution of states, against single-spin flip simulations as the basline.

---

## Overview

The package relies on mostly standard Julia libraries, and uses **FLUX.jl** for machine learning implementations.

The `\experiments` directory contains the scripts used to generate data and train and validate the models. Additionally, the `experiments\experiment1.md` file serves as an explaination of these scripts, and a ***quickstart guide*** for user to run their own experiments. The scripts are general enough for a user to be able to generate results very quickly, requiring only a choice of some system parameters.

For adaptability, the code in `\src\ML.jl` is what users should seek to modify, changing the structure of training data, target computation, loss fucntion, and model and optimizer definitions as they like. This is independent of the architecture to run-simulations (both for data generation and model validation), allowing users to swap components without rewriting entire simulations.

-- -

## Installation

Clone the Repository, and activate the required dependencies
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```


