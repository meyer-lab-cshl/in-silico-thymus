## thymus_ABM

Implementation of agent-based model to simulate negative selection in the thymus.

### Requirements
Base

Agents

Random

InteractiveDynamics

GLMakie

Statistics

StatsBase

DelimitedFiles

NamedArrays

NPZ

JSON

CSV

ArgParse

### Usage

`thymus_abm_3D.jl` is a Julia script that runs simulations specified by command line arguments.

`$ julia thymus_abm_3D.jl --steps 5000 --n_tecs 500 --n_dendritics 50 --n_thymocytes 5000 --selection_threshold 1.6 --dt 1.0 --rng 42 --synapse_interactions 5 --min_strong_interactions 1 --dim1 10.0 --dim2 10.0 --dim3 10.0`
