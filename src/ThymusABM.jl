module ThymusABM

export Tec, Dendritic, Thymocyte, Parameters, add_tecs!, add_dendritics!, add_thymocytes!, calculate_reaction_strength, initialize, cell_move!, set_color!, tec_DC_interact!, thymocyte_APC_interact!, update_tec_stage, collide!, model_step!, cell_colors, cell_sizes, cell_markers, parse_commandline, tec, thymocyte, react_ratio, escape_ratio, escapedautoreactive_ratio, nonreact_ratio, total_thy, alive_ratio
#using Distributed
#addprocs(2)
#@everywhere begin
using Base: Float64
using Agents: length, isempty, getindex
using Agents
using Random
using InteractiveDynamics
using Statistics: mean
using StatsBase # used to sample without replacement
using DelimitedFiles
using NamedArrays
using NPZ
using JSON
using CSV
using ArgParse
using LinearAlgebra
using Mmap

################## optimize checking of all peptides for escaped autoreactives #################
# make Set of only genes/peptides actually present in simulation?
# or keep it to be entire .txt file? - this is how it is currently
################## check tec_DC interaction ####################################################
# how to transfer peptides?
# all or only some? - transfer all for now
"""
    Tec
    id::Int                             # Unique ID to identify agent
    pos::NTuple{3,Float64}              # 2D position of agent
    vel::NTuple{3,Float64}              # 2D velocity of agent
    mass::Float64                       # Mass of agent to use in collisions
    type::Symbol                        # Cell type of agent
    color::String                       # Color used for agent in videos
    size::Float64                       # Size of agent in videos
    num_interactions::Int               # Total number of interactions agent has
    antigens::Array                     # List of antigens the Tec agent contains
    stage::Int                          # Maturation stage of Tec
    steps_alive::Int                    # Number of steps the Tec agent has been alive for

Tec agent struct.
"""
mutable struct Tec <: AbstractAgent
    id::Int                             # Unique ID to identify agent
    pos::NTuple{3,Float64}              # 2D position of agent
    vel::NTuple{3,Float64}              # 2D velocity of agent
    mass::Float64                       # Mass of agent to use in collisions
    type::Symbol                        # Cell type of agent
    color::String                       # Color used for agent in videos
    size::Float64                       # Size of agent in videos
    num_interactions::Int               # Total number of interactions agent has
    antigens::Array                     # List of antigens the Tec agent contains
    stage::Int                          # Maturation stage of Tec
    steps_alive::Int
end

"""
    Dendritic
    id::Int                             # Unique ID to identify agent
    pos::NTuple{3,Float64}              # 2D position of agent
    vel::NTuple{3,Float64}              # 2D velocity of agent
    mass::Float64                       # Mass of agent to use in collisions
    type::Symbol                        # Cell type of agent
    color::String                       # Color used for agent in videos
    size::Float64                       # Size of agent in videos
    num_interactions::Int               # Total number of interactions agent has
    antigens::Array                     # List of antigens the Dendritic agent contains
    steps_alive::Int                    # Number of steps the Dendritic agent has been alive for

Dendritic agent struct.
"""
mutable struct Dendritic <: AbstractAgent
    id::Int                             # Unique ID to identify agent
    pos::NTuple{3,Float64}              # 2D position of agent
    vel::NTuple{3,Float64}              # 2D velocity of agent
    mass::Float64                       # Mass of agent to use in collisions
    type::Symbol                        # Cell type of agent
    color::String                       # Color used for agent in videos
    size::Float64                       # Size of agent in videos
    num_interactions::Int               # Total number of interactions agent has
    antigens::Array                     # List of antigens the Dendritic agent contains
    steps_alive::Int
end

"""
    Thymocyte
    id::Int                             # Unique ID to identify agent
    pos::NTuple{3,Float64}              # 2D position of agent
    vel::NTuple{3,Float64}              # 2D velocity of agent
    mass::Float64                       # Mass of agent to use in collisions
    type::Symbol                        # Cell type of agent
    color::String                       # Color used for agent in videos
    size::Float64                       # Size of agent in videos
    num_interactions::Int               # Total number of interactions agent has
    tcr::String                         # TCR that the thymocyte agent is carrying
    reaction_levels::Dict               # Dict to hold thymocyte's seen antigens and reaction levels to them
    death_label::Bool                   # Label a thymocyte to be killed on its next step
    confined::Bool                      # Boolean if the thymocyte has been confined by an APC
    bind_location::NTuple{3,Float64}    # Location of binding w/ APC that led to confinement
    treg::Bool                          # Boolean if thymocyte meets Treg threshold or not
    steps_alive::Int                    # Number of steps the Thymocyte agent has been alive for

Thymocyte agent struct.
"""
mutable struct Thymocyte <: AbstractAgent
    id::Int                             # Unique ID to identify agent
    pos::NTuple{3,Float64}              # 2D position of agent
    vel::NTuple{3,Float64}              # 2D velocity of agent
    mass::Float64                       # Mass of agent to use in collisions
    type::Symbol                        # Cell type of agent
    color::String                       # Color used for agent in videos
    size::Float64                       # Size of agent in videos
    num_interactions::Int               # Total number of interactions agent has
    tcr::String                         # TCR that the thymocyte agent is carrying
    reaction_levels::Dict               # Dict to hold thymocyte's seen antigens and reaction levels to them
    death_label::Bool                   # Label a thymocyte to be killed on its next step
    confined::Bool                      # Boolean if the thymocyte has been confined by an APC
    bind_location::NTuple{3,Float64}    # Location of binding w/ APC that led to confinement
    treg::Bool                          # Boolean if thymocyte meets Treg threshold or not
    steps_alive::Int
end

"""
    Parameters
    width_height::NTuple{3,Float64} = width_height                          # 3-Tuple of Float64 for 3 dimensions of the simulation (x, y, z)
    speed::Float64 = speed                                                  # Float64 speed factor
    dt::Float64 = dt                                                        # Float64 of time step governing movement/velocity of agents
    n_tecs::Int = n_tecs                                                    # Int of number of mTECs to have in simulation
    n_thymocytes::Int = n_thymocytes                                        # Int of number of thymocytes to have in simulation
    n_dendritics::Int = n_dendritics                                        # Int of number of DCs to have in simulation
    threshold::Int16 = threshold                                            # Int of negative selection threshold of thymocytes
    peptides::Array = peptides                                              # Array of all possible peptides a mTEC/DC can present
    tcrs::Array = tcrs                                                      # Array of generated TCRs for thymocyte to randomly select from
    successful_interactions::Int = successful_interactions                  # Int of number of times a thymocyte was selected in an interaction with mTEC/DC
    unsuccessful_interactions::Int = unsuccessful_interactions              # Int of number of times a thymocyte was not selected in an interaction with mTEC/DC
    escaped_thymocytes::Int = 0                                             # Int of number of thymocytes that were not negatively selected that should have been
    autoreactive_thymocytes::Int = 0                                        # Int of number of thymocytes that were negatively selected or escaped
    nonautoreactive_thymocytes::Int = 0                                     # Int of number of thymocytes that were not autoreactive and can be assumed to have successfully exited the thymus
    total_thymocytes::Int = 0                                               # Int of total number of thymocytes that have entered simulation
    treg_threshold::Float64 = treg_threshold                                # Float64 setting the threshold for a nonautoreactive thymocyte to be classified as a Treg
    num_tregs::Int = 0                                                      # Int of total number of Tregs that have successfully exited simulation
    tecs_present::Int = n_tecs                                              # Int of number of mTECs currently present in the simulation
    aa_matrix::NamedArray = aa_matrix                                       # NamedArray containing all possible interaction strengths between peptides and tcrs
    synapse_interactions::Int = synapse_interactions                        # Int of total number of peptide:TCR reactions to calculate for one thymocyte:mTEC/DC interaction
    min_strong_interactions::Int = min_strong_interactions                  # Int of minimum number of reactions in a thymocyte:mTEC/DC interaction that need to be strong enough to cause negative selection         
    deaths::Int = deaths                                                    # Int of total number of thymocytes that have exited simulation for any reason at current step
    total_dead_thymocytes::Int = total_dead_thymocytes                      # Int of total number of thymocytes that have exited simulation for any reason across all steps
    alive_thymocytes::Int = alive_thymocytes                                # Int of total number of alive thymocytes at current step
    stage_genes_peptides_dict::Vector{Any} = stage_genes_peptides_dict      # Vector holding the development stage information for mTECs (genes and corresponding peptides for each stage)
    max_tec_interactions::Int = max_tec_interactions                        # Int of maximum number of interactions a mTEC can have before it "dies" and is replaced by a new one
    max_thymocyte_interactions::Int = max_thymocyte_interactions            # Int of maximum number of interactions a thymocyte must have before it leaves as a nonautoreactive T cell
    step::Int = 0                                                           # Int of current step of simulation

Parameters struct.
"""
Base.@kwdef mutable struct Parameters
    width_height::NTuple{3,Float64} = width_height                          # 3-Tuple of Float64 for 3 dimensions of the simulation (x, y, z)
    speed::Float64 = speed                                                  # Float64 speed factor
    dt::Float64 = dt                                                        # Float64 of time step governing movement/velocity of agents
    n_tecs::Int = n_tecs                                                    # Int of number of mTECs to have in simulation
    n_thymocytes::Int = n_thymocytes                                        # Int of number of thymocytes to have in simulation
    n_dendritics::Int = n_dendritics                                        # Int of number of DCs to have in simulation
    threshold::Int16 = threshold                                            # Int of negative selection threshold of thymocytes
    peptides::Array = peptides                                              # Array of all possible peptides a mTEC/DC can present
    tcrs::Array = tcrs                                                      # Array of generated TCRs for thymocyte to randomly select from
    successful_interactions::Int = successful_interactions                  # Int of number of times a thymocyte was selected in an interaction with mTEC/DC
    unsuccessful_interactions::Int = unsuccessful_interactions              # Int of number of times a thymocyte was not selected in an interaction with mTEC/DC
    escaped_thymocytes::Int = 0                                             # Int of number of thymocytes that were not negatively selected that should have been
    autoreactive_thymocytes::Int = 0                                        # Int of number of thymocytes that were negatively selected or escaped
    nonautoreactive_thymocytes::Int = 0                                     # Int of number of thymocytes that were not autoreactive and can be assumed to have successfully exited the thymus
    total_thymocytes::Int = 0                                               # Int of total number of thymocytes that have entered simulation
    treg_threshold::Float64 = treg_threshold                                # Float64 setting the threshold for a nonautoreactive thymocyte to be classified as a Treg
    num_tregs::Int = 0                                                      # Int of total number of Tregs that have successfully exited simulation
    tecs_present::Int = n_tecs                                              # Int of number of mTECs currently present in the simulation
    aa_matrix::NamedArray = aa_matrix                                       # NamedArray containing all possible interaction strengths between peptides and tcrs
    synapse_interactions::Int = synapse_interactions                        # Int of total number of peptide:TCR reactions to calculate for one thymocyte:mTEC/DC interaction
    min_strong_interactions::Int = min_strong_interactions                  # Int of minimum number of reactions in a thymocyte:mTEC/DC interaction that need to be strong enough to cause negative selection         
    deaths::Int = deaths                                                    # Int of total number of thymocytes that have exited simulation for any reason at current step
    total_dead_thymocytes::Int = total_dead_thymocytes                      # Int of total number of thymocytes that have exited simulation for any reason across all steps
    alive_thymocytes::Int = alive_thymocytes                                # Int of total number of alive thymocytes at current step
    stage_genes_peptides_dict::Vector{Any} = stage_genes_peptides_dict      # Vector holding the development stage information for mTECs (genes and corresponding peptides for each stage)
    max_tec_interactions::Int = max_tec_interactions                        # Int of maximum number of interactions a mTEC can have before it "dies" and is replaced by a new one
    max_thymocyte_interactions::Int = max_thymocyte_interactions            # Int of maximum number of interactions a thymocyte must have before it leaves as a nonautoreactive T cell
    step::Int = 0                                                           # Int of current step of simulation
end

"""
    add_tecs!(model, n_tecs, color, size, replenishing)

Adds `n_tecs` number of Tecs to the `model` of given `color` and `size`. Boolean `replenishing` determines if Tec is an initial Tec (true), or added to the model later on (false).
"""
function add_tecs!(model, n_tecs, color, size, replenishing)
    for _ in 1:n_tecs
        id = nextid(model)
        pos = Tuple(rand(model.rng, 3))
        velocity = (0.0, 0.0, 0.0)
        mass = Inf
        steps_alive = 0
        #if !isempty(nearby_ids(pos, model, model.width_height[1]/2)) # make sure mTECs don't overlap
        #    pos = Tuple(rand(model.rng, 3))
        #end
        if replenishing == false
            num_interactions = rand(model.rng, 0:model.max_tec_interactions - 1)
            stage = rand(model.rng, 1:length(model.stage_genes_peptides_dict))
        else
            num_interactions = 0
            stage = 1
        end

        valid_genes_peptides = model.stage_genes_peptides_dict[stage]
        # how many genes for 1 mTEC? do we explicitly care about keeping gene names, or just combine their peptides into 1 array?
        # check rng
        genes = rand(model.rng, valid_genes_peptides, 100)
        peptides = vcat([gene[2] for gene in genes]...) # since gene is a gene -> peptides pair, gene[2] = peptides for that gene
        #antis = sample(model.rng, model.peptides, num_interactions+100) # choose a size sample from peptides to act as tec's antigens
        tec = Tec(id, pos, velocity, mass, :tec, color, size, num_interactions, peptides, stage, steps_alive)
        add_agent!(tec, model)
        set_color!(tec, model)
    end
end

"""
    add_dendritics!(model, n_dendritics, color, size)

Adds `n_dendritics` number of Dendritics to the `model` of given `color` and `size`.
"""
function add_dendritics!(model, n_dendritics, color, size)
    # review DC peptides - how many they start with, how they gain new ones
    for _ in 1:n_dendritics
        id = nextid(model)
        pos = Tuple(rand(model.rng, 3))
        #if !isempty(nearby_ids(pos, model, model.width_height[1]/2)) # make sure mTECs/dendritics don't overlap
        #    pos = Tuple(rand(model.rng, 3))
        #end
        velocity = ((sincos(2π * rand(model.rng)) .* model.speed)...,sin(2π * rand(model.rng)) .* model.speed)
        mass = 1.0
        steps_alive = 0
        num_interactions = rand(model.rng, 0:model.max_tec_interactions - 1)

        stage = rand(model.rng, 1:length(model.stage_genes_peptides_dict)) # figure out best way to increment stage
        valid_genes_peptides = model.stage_genes_peptides_dict[stage]
        # how many genes for 1 mTEC? do we explicitly care about keeping gene names, or just combine their peptides into 1 array?
        # check rng
        genes = rand(model.rng, valid_genes_peptides, 100)
        peptides = vcat([gene[2] for gene in genes]...) # since gene is a gene -> peptides pair, gene[2] = peptides for that gene

        #antis = sample(model.rng, model.peptides, 1, replace = false) # choose 1 antigen for DC to start with
        dc = Dendritic(id, pos, velocity, mass, :dendritic, color, size, num_interactions, peptides, steps_alive)
        add_agent!(dc, model)
        #set_color!(dc)
    end
end

"""
    add_thymocytes!(model, n_thymocytes, color, size, initial)

Adds `n_thymocytes` number of Thymocytes to the `model` of given `color` and `size`. Boolean `initial` determines if Thymocyte is an initial Thymocyte (true), or added to the model later on (false).
"""
function add_thymocytes!(model, n_thymocytes, color, size, initial)
    for _ in 1:n_thymocytes
        id = nextid(model)
        model.total_thymocytes += 1
        model.alive_thymocytes += 1
        pos = Tuple(rand(model.rng, 3))
        vel = ((sincos(2π * rand(model.rng)) .* model.speed)...,sin(2π * rand(model.rng)) .* model.speed)
        mass = 1.0
        #tcr = randstring(model.rng, "ACDEFGHIKLMNPQRSTVWY", 9)
        tcr = rand(model.rng, model.tcrs)
        steps_alive = 0
        if initial == true
            num_interactions = rand(model.rng, 0:model.max_thymocyte_interactions - 1) #if want to randomize this, also have to randomize initial reaction levels
            reaction_levels = Dict{String, Float32}()
            if num_interactions != 0
                for i in 1:num_interactions
                    pept = rand(model.rng, model.peptides)
                    calculate_reaction_strength(model, pept, tcr, reaction_levels)
                end
            end
        else
            num_interactions = 0
            reaction_levels = Dict{String, Float32}()
        end

        death_label = false
        confined = false
        bind_location = (0.0,0.0,0.0)
        treg = false
        thymocyte = Thymocyte(id, pos, vel, mass, :thymocyte, color, size, num_interactions, tcr, reaction_levels, death_label, confined, bind_location, treg, steps_alive)
        add_agent!(thymocyte, model)
        #set_color!(thymocyte)
    end
end

"""
    calculate_reaction_strength(model, peptide, tcr, reaction_levels)

Calculate the strength of the interaction between given `peptide` and `tcr` for the `model`. Store the calculated strength in the Thymocyte's `reaction_levels` dictionary.
"""
function calculate_reaction_strength(model, peptide,  tcr, reaction_levels)
#=     reaction = 1.0
    for i in range(1, length(peptide), step=1)
        antigen_aa = peptide[i]
        tcr_aa = tcr[i]
        reaction *= model.aa_matrix[i, tcr_aa, antigen_aa]
    end
    reaction = reaction^(1/length(peptide)) =#
    reaction = model.aa_matrix[peptide, tcr]
    if reaction > 100
        reaction = 100
    end
    if get(reaction_levels, peptide, 0) != 0 # if thymocyte has seen antigen before, add to its current reaction level
        reaction_levels[peptide] += reaction
    else # otherwise, add antigen as a new entry to the reaction_levels dict
        reaction_levels[peptide] = reaction
    end
end

"""
    initialize(;
        width_height = (1.0, 1.0, 1.0),
        speed = 0.002,
        n_tecs = 50,
        n_thymocytes = 1000,
        n_dendritics = 50,
        dt=1.0,
        threshold = 0.8,
        treg_threshold = 0.6,
        rng_seed = 1,
        synapse_interactions = 1,
        min_strong_interactions = 1,
        max_tec_interactions = 200,
        max_thymocyte_interactions = 80)

Initialize the model with default or specified parameters. Returns the initialized `model`.
"""
function initialize(;
    width_height = (1.0, 1.0, 1.0),
    speed = 0.002,
    n_tecs = 50,
    n_thymocytes = 1000,
    n_dendritics = 50,
    dt=1.0,
    threshold = 0.8,
    treg_threshold = 0.6,
    rng_seed = 1,
    synapse_interactions = 1,
    min_strong_interactions = 1,
    max_tec_interactions = 200,
    max_thymocyte_interactions = 80)

    rng = MersenneTwister(rng_seed)
    step = 0

#=     possible_antigens = readdlm("/home/mulle/Documents/JuliaFiles/thymus_ABM/data/validpeptides.txt",'\n')
    #peptides = sample(rng, possible_antigens, replace=false)
    peptides = unique(vec(possible_antigens)) =#

    stage_genes_peptides_dict = JSON.parsefile("./data/stage_genes_peptides.json")

    space3d = ContinuousSpace(width_height, 1.0) # change number here depending on volume dimensions used

    successful_interactions = 0
    unsuccessful_interactions = 0
    autoreactive_thymocytes = 0
    nonautoreactive_thymocytes = 0
    escaped_thymocytes = 0
    deaths = 0
    total_dead_thymocytes = 0
    alive_thymocytes = 0

    total_thymocytes = 0
    num_tregs = 0

    #aa_data = npzread("/home/mulle/Documents/JuliaFiles/thymus_ABM/data/H2_proportional_binding_matrices.npy")
    #aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    pepts = readlines(open("./data/uniquepeptides.txt"))
    tcrs = readlines(open("./data/tcrs10000.txt"))
    matches = open("./data/matches.dist")
    matr = Mmap.mmap(matches, Matrix{Float32}, (size(pepts)[1], size(tcrs)[1]))
    aa_matrix = NamedArray(matr, (pepts, tcrs), ("pepts", "TCRs"))
    close(matches)

    properties = Parameters(width_height, speed, dt, n_tecs, n_thymocytes, n_dendritics, threshold, pepts, tcrs, successful_interactions, unsuccessful_interactions, escaped_thymocytes, autoreactive_thymocytes, nonautoreactive_thymocytes,
        total_thymocytes, treg_threshold, num_tregs, n_tecs, aa_matrix, synapse_interactions, min_strong_interactions, deaths, total_dead_thymocytes, alive_thymocytes, stage_genes_peptides_dict, max_tec_interactions,
        max_thymocyte_interactions, step)
    
    model = ABM(Union{Tec, Dendritic, Thymocyte}, space3d; properties, rng,)

    # Add agents to the model
    add_tecs!(model, n_tecs, "#00ffff", 0.5, false)
    add_dendritics!(model, n_dendritics, "#ffa500", 0.5)
    add_thymocytes!(model, n_thymocytes, "#edf8e9", 0.2, true)

    return model
end

## Agent steps
"""
    cell_move!(agent::Union{Tec, Dendritic, Thymocyte}, model)

Move the given `agent` of the `model` according to its type.
"""
function cell_move!(agent::Union{Tec, Dendritic, Thymocyte}, model)
    if agent.type == :thymocyte
        if agent.death_label == true # maybe weird to take care of agent death here, but doing it in interact! in model_step! sometimes causes key value errors - does this introduce any problems?
            kill_agent!(agent, model)
            model.autoreactive_thymocytes += 1
            model.deaths += 1
            model.total_dead_thymocytes += 1
            return
            # Fix movement under confinement below? - some agents move back and forth over short distance - confine around location that thymocyte binded or location of binding tec or is that the same thing?
        elseif agent.confined == true
            if agent.pos[1] >= agent.bind_location[1] + 0.3 || agent.pos[2] >= agent.bind_location[2] + 0.3 || agent.pos[1] >= agent.bind_location[1] - 0.3 || agent.pos[2] >= agent.bind_location[2] - 0.3 ||
                agent.pos[3] >= agent.bind_location[3] + 0.3 || agent.pos[3] >= agent.bind_location[3] - 0.3
                if get_direction(agent.pos, agent.bind_location, model)[1] < 0 || get_direction(agent.pos, agent.bind_location, model)[2] < 0 || get_direction(agent.pos, agent.bind_location, model)[3] < 0
                    agent.vel = -1 .* agent.vel
                end
            end
        end
    end
    move_agent!(agent, model, model.dt)
    set_color!(agent, model)
end

"""
    set_color!(agent::Union{Tec, Dendritic, Thymocyte}, model)

Set the color of the given `agent` of the `model` according to its type.
"""
function set_color!(agent::Union{Tec, Dendritic, Thymocyte}, model)
    if agent.type == :tec
        colors = ["#f7fbff","#deebf7","#c6dbef","#9ecae1","#6baed6","#4292c6","#2171b5","#08519c","#08306b"]
        if agent.stage <= length(model.stage_genes_peptides_dict)
            agent.color = colors[agent.stage]
        end
    elseif agent.type == :thymocyte
        if agent.confined == true
            agent.color = "#ffff00"
        elseif any(x -> x <= model.threshold / 4, values(agent.reaction_levels))
            agent.color = "#bae4b3"
        elseif any(x -> x <= model.threshold / 2, values(agent.reaction_levels))
            agent.color = "#74c476"
        elseif any(x -> x <= Inf, values(agent.reaction_levels))
            agent.color = "#238b45"
        else
            agent.color = "#edf8e9"
        end
    else
        return
    end
end

## Model steps
"""
    tec_DC_interact!(a1::Union{Tec, Dendritic, Thymocyte}, a2::Union{Tec, Dendritic, Thymocyte}, model)

Governs the interaction between a Tec and Dendritic of the `model` who come into contact. Transfers Tec antigens to Dendritic antigens.
"""
function tec_DC_interact!(a1::Union{Tec, Dendritic, Thymocyte}, a2::Union{Tec, Dendritic, Thymocyte}, model)
    if a1.type == :tec && a2.type == :dendritic # if tec/thymocyte collide, they can interact. relabel them here for simplicity
        tec_agent = a1
        dendritic_agent = a2
    elseif a1.type == :dendritic && a2.type == :tec
        tec_agent = a2
        dendritic_agent = a1
    else
        return
    end

    #peptides = rand(model.rng, tec_agent.antigens, model.synapse_interactions)

    #push!(dendritic_agent.antigens, peptides...)#rand(model.rng, model.peptides))
    dendritic_agent.antigens = tec_agent.antigens
end

"""
    thymocyte_APC_interact!(a1::Union{Tec, Dendritic, Thymocyte}, a2::Union{Tec, Dendritic, Thymocyte}, model)

Governs the interaction between a Thymocyte and Tec (or Dendritic) of the `model` who come into contact. Calculates Thymocyte TCR binding affinity to one of Tec or Dendritic's peptides.
"""
function thymocyte_APC_interact!(a1::Union{Tec, Dendritic, Thymocyte}, a2::Union{Tec, Dendritic, Thymocyte}, model)
    if (a1.type == :tec || a1.type == :dendritic) && a2.type == :thymocyte # if tec/thymocyte collide, they can interact. relabel them here for simplicity
        tec_agent = a1
        thymocyte_agent = a2
    elseif a1.type == :thymocyte && (a2.type == :tec || a2.type == :dendritic)
        tec_agent = a2
        thymocyte_agent = a1
    else
        return
    end

    # compare a chosen tec antigen sequence to thymocyte TCR sequence
    # choose random antigen from tec's antigens to compare thymocyte tcr to. use aa_matrix to retrieve stength of interaction, comparing characters one by one
    # if a reaction level passes the model threshold, the thymocyte is killed
    antigens = rand(model.rng, tec_agent.antigens, model.synapse_interactions)
    
    total_strength = 0
    strong_reactions = 0
    for antigen in antigens
        calculate_reaction_strength(model, antigen, thymocyte_agent.tcr, thymocyte_agent.reaction_levels)
        #thymocyte_agent.reaction_levels[antigen] = min(thymocyte_agent.reaction_levels[antigen], 1.0) # clip reaction strength to max of 1.0
        total_strength += thymocyte_agent.reaction_levels[antigen]

        if thymocyte_agent.reaction_levels[antigen] >= model.threshold
            strong_reactions += 1
        else
            if thymocyte_agent.reaction_levels[antigen] >= model.treg_threshold && thymocyte_agent.reaction_levels[antigen] < model.threshold
                thymocyte_agent.treg = true
            end
            model.unsuccessful_interactions += 1
        end

        if strong_reactions >= model.min_strong_interactions || total_strength >= model.threshold # kill thymocyte if sequence matches are above model threshold - (> or >=?) or if total strength from multiple interacts is above thresh
            #kill_agent!(thymocyte_agent, model)
            if rand(model.rng) > 0.3
                thymocyte_agent.death_label = true
            else
                if thymocyte_agent.confined == false
                    thymocyte_agent.confined = true
                    thymocyte_agent.bind_location = thymocyte_agent.pos
                    thymocyte_agent.vel = 0.5 .* thymocyte_agent.vel
                end
            end
            model.successful_interactions += 1
            break
        end
    end
    # change stages here?
    thymocyte_agent.num_interactions += 1
    tec_agent.num_interactions += 1 
    update_tec_stage(tec_agent, model)

    set_color!(a1, model)
    set_color!(a2, model)
end

"""
    update_tec_stage(tec, model)

Updates the development stage of the given Tec agent `tec` of the `model` according to how many interactions with Thymocytes it has had so far.
"""
function update_tec_stage(tec, model)
    if tec.num_interactions >= model.max_tec_interactions / length(model.stage_genes_peptides_dict) && tec.type == :tec
        tec.stage += 1
        if tec.stage <= length(model.stage_genes_peptides_dict)
            valid_genes_peptides = model.stage_genes_peptides_dict[tec.stage]
            genes = rand(model.rng, valid_genes_peptides, 100)
            tec.antigens  = vcat([gene[2] for gene in genes]...)
            tec.num_interactions = 0
        end
    end
end

"""
    collide!(a::Union{Tec, Dendritic, Thymocyte}, b::Union{Tec, Dendritic, Thymocyte})

Calculates an elastic collision between two agents, `a` and `b`, when they collide with each other. Adapted from Agents.jl elastic_collision and modified to work for 3D animations.
"""
function collide!(a::Union{Tec, Dendritic, Thymocyte}, b::Union{Tec, Dendritic, Thymocyte})
    # using http://www.hakenberg.de/diffgeo/collision_resolution.htm without any angular information
    # Same check to prevent double collisions from Agents.jl elastic_collision function
    v1, v2, x1, x2 = a.vel, b.vel, a.pos, b.pos
    r1 = x1 .- x2
    r2 = x2 .- x1
    m1, m2 = a.mass, b.mass
    m1 == m2 == Inf && return false
    if m1 == Inf
        dot(r1, v2) ≤ 0 && return false
    elseif m2 == Inf
        dot(r2, v1) ≤ 0 && return false
    else
        !(dot(r2, v1) > 0 && dot(r2, v1) > 0) && return false
    end

    # Calculate results of elastic collision
    n = (a.pos .- b.pos) ./ (sqrt((a.pos[1] - b.pos[1])^2 + (a.pos[2] - b.pos[2])^2 + (a.pos[3] - b.pos[3])^2)) # unit normal vector
    λ = 2 .* ((dot(a.vel, n) - dot(b.vel, n)) / (dot((1/a.mass + 1/b.mass) .* n, n))) # lambda parameter from website
    a.vel = a.vel .- (λ/a.mass).*n # update velocity a
    b.vel = b.vel .+ (λ/b.mass).*n # update velocity b
end

"""
    model_step!(model)

Steps the `model` one step forward. Calculates all agent interactions for the given step. Determines when an agent leaves due to age and adds new agent to replenish it.
"""
function model_step!(model) # happens after every agent has acted
    model.step += 1
    if model.step % 100 == 0
        println(model.step)
    end

    interaction_radius = 0.06*model.width_height[1]
    for (a1, a2) in interacting_pairs(model, interaction_radius, :types) # check :all versus :nearest versus :types.
        #:types allows for easy changing of tec size by changing radius, but then thymocytes do not interact at all. :all causes error if thymocyte is deleted while having > 1 interaction. :nearest only allows for 1 interaction 
        tec_DC_interact!(a1, a2, model)
        thymocyte_APC_interact!(a1, a2, model)
        #elastic_collision!(a1, a2, :mass)
        collide!(a1, a2)
    end

    for agent in allagents(model)
        agent.steps_alive += 1
        escaped = false
        if (agent.type == :tec && agent.stage >= length(model.stage_genes_peptides_dict) + 1) || (agent.num_interactions >= model.max_thymocyte_interactions && agent.type == :thymocyte)
            if agent.type == :tec
                model.tecs_present -= 1 
            elseif agent.type == :thymocyte
                model.deaths += 1
                model.total_dead_thymocytes += 1
                strong_reactions = 0
                for antigen in model.peptides # check if exiting thymocyte was autoreactive by comparing its TCR to every peptide possible to be presented
#=                     reaction = 1.0
                    for i in range(1, length(antigen), step=1)
                        antigen_aa = antigen[i]
                        tcr_aa = agent.tcr[i]
                        reaction *= model.aa_matrix[i, tcr_aa, antigen_aa]
                    end
                    reaction = reaction^(1/length(antigen)) =#
                    reaction = model.aa_matrix[antigen, agent.tcr]
                    if reaction > 100
                        reaction = 100
                    end
#=                     if get(agent.reaction_levels, antigen, 0) != 0 # if thymocyte has seen antigen before, add to its current reaction level
                        agent.reaction_levels[antigen] += reaction
                    else # otherwise, add antigen as a new entry to the reaction_levels dict
                        agent.reaction_levels[antigen] = reaction
                    end =#
                    if reaction >= model.threshold
                        strong_reactions += 1
                    end

                    if strong_reactions >= model.min_strong_interactions
                        model.autoreactive_thymocytes += 1
                        model.escaped_thymocytes += 1
                        escaped = true
                        break
                    end
                end

                if agent.treg == true
                    model.num_tregs += 1
                end

                if escaped == false
                    model.nonautoreactive_thymocytes += 1
                end
            end
            kill_agent!(agent, model)
        end
    end

    if model.tecs_present < model.n_tecs # generate new tecs if some died
        tecs_missing = model.n_tecs - model.tecs_present
        add_tecs!(model, tecs_missing, "#00c8ff", 0.5, true)
        model.tecs_present += tecs_missing
    end

    add_thymocytes!(model, model.deaths, "#edf8e9", 0.2, false) # replenish thymocytes
    model.deaths = 0

    model.alive_thymocytes = count(i->(i.type == :thymocyte), allagents(model))
end

"""
    cell_colors(a)

Return the color of the given agent `a`.
"""
cell_colors(a) = a.color

"""
    cell_sizes(a)

Return the size of the given agent `a`.
"""
cell_sizes(a) = a.size

"""
    cell_markers(a)

Return the marker of the given agent `a`.
"""
function cell_markers(a::Union{Tec, Dendritic, Thymocyte})
    if a.type == :thymocyte
        return :circle
    elseif a.type == :tec
        return :star5
    else
        return :diamond
    end
end

"""
    parse_commandline()

Parse command line arguments to set up the ABM.
"""
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--steps"
            help = "Help"
            arg_type = Int
            default = 1000
        "--dim1"
            help = "an option with an argument"
            arg_type = Float64
            default = 10.0
        "--dim2"
            help = "an option with an argument"
            arg_type = Float64
            default = 10.0
        "--dim3"
            help = "an option with an argument"
            arg_type = Float64
            default = 10.0
        "--n_tecs"
            help = "another option with an argument"
            arg_type = Int
            default = 500
        "--n_dendritics"
            help = "Help"
            arg_type = Int
            default = 50
        "--n_thymocytes"
            help = "Help"
            arg_type = Int
            default = 5000
        "--selection_threshold"
            help = "Help"
            arg_type = Float64
            default = 100.0
        "--dt"
            help = "Help"
            arg_type = Float64
            default = 1.0
        "--rng"
            help = "Help"
            arg_type = Int
            default = 1
        "--synapse_interactions"
            help = "Help"
            arg_type = Int
            default = 1
        "--min_strong_interactions"
            help = "Help"
            arg_type = Int
            default = 1
    end

    return parse_args(s)
end

"""
    tec(a)

Used to check if agent `a` is a Tec.
"""
tec(a) = a.type == :tec

"""
    thymocyte(a)

Used to check if agent `a` is a Thymocyte.
"""
thymocyte(a) = a.type == :thymocyte

global adata = [(thymocyte, count), (tec, count)]
global alabels = ["Alive Thymocytes", "Alive mTECs"]

"""
    react_ratio(model)

Return the proportion of exited and dead thymocytes that were autoreactive in the `model`.
"""
react_ratio(model) = model.autoreactive_thymocytes/model.total_dead_thymocytes # proportion of total killed/exited thymocytes that were autoreactive

"""
    escape_ratio(model)

Return the proportion of exited and dead thymocytes that escaped in the `model`.
"""
escape_ratio(model) = model.escaped_thymocytes/model.total_dead_thymocytes  # proportion of total killed/exited thymocytes that escaped

"""
    escapedautoreactive_ratio(model)

Return the proportion of autoreactive thymocytes that escaped in the `model`.
"""
escapedautoreactive_ratio(model) = model.escaped_thymocytes/model.autoreactive_thymocytes # proportion of total autoreactive thymocytes that escaped

"""
    nonreact_ratio(model)

Return the proportion of exited and dead thymocytes that were non-autoreactive in the `model`.
"""
nonreact_ratio(model) = model.nonautoreactive_thymocytes/model.total_dead_thymocytes  # proportion of total killed/exited thymocytes that were nonautoreactive

"""
    total_thy(model)

Return the total number of dead, exited, and alive thymocytes by end of simulation of the `model`.
"""
total_thy(model) = model.total_thymocytes

"""
    alive_ratio(model)

Return the proportion of alive thymocytes to exited and dead thymocytes in the `model`.
"""
alive_ratio(model) = model.alive_thymocytes/model.total_dead_thymocytes


global mdata = [:num_tregs, :autoreactive_thymocytes, :escaped_thymocytes, :nonautoreactive_thymocytes, :alive_thymocytes, escape_ratio, react_ratio, nonreact_ratio, :threshold, total_thy, alive_ratio, escapedautoreactive_ratio]
global mlabels = ["number of tregs", "autoreactive", "escaped", "nonautoreactive", "alive", "escaped thymocytes ratio", "autoreactive thymocytes ratio", "nonautoreactive ratio", "selection threshold", "total thymocytes", "alive thymocytes ratio", "escaped to autoreactive ratio"]
#mdata = [:autoreactive_thymocytes, :nonautoreactive_thymocytes, :escaped_thymocytes]
#mlabels = ["Autoreactive Thymocytes", "Non-autoreactive Thymocytes", "Escaped Autoreactive Thymocytes"]

global dims = (10.0, 10.0, 10.0) # seems to work best for 3D
global agent_speed = 0.0015 * dims[1]
#model2 = initialize(; width_height = dims, n_tecs = 500, n_dendritics = 50, n_thymocytes = 5000, speed = agent_speed, threshold = 1.6, dt = 1.0, rng_seed = 42, treg_threshold = 0.6, synapse_interactions = 3, min_strong_interactions = 1,
#)
#= parsed_args = parse_commandline()
model2 = initialize(; width_height = tuple(parsed_args["dim1"], parsed_args["dim2"], parsed_args["dim3"]), n_tecs = parsed_args["n_tecs"], n_dendritics = parsed_args["n_dendritics"], 
n_thymocytes = parsed_args["n_thymocytes"], speed = agent_speed, threshold = parsed_args["selection_threshold"], dt = parsed_args["dt"], rng_seed = parsed_args["rng"], treg_threshold = 0.6, 
synapse_interactions = parsed_args["synapse_interactions"], min_strong_interactions = parsed_args["min_strong_interactions"])

#global parange = Dict(:threshold => 0:0.01:1)

@time adf, mdf = run!(model2, cell_move!, model_step!, 1000; adata = adata, mdata = mdata)
CSV.write("./data/adf.csv", adf)
CSV.write("./data/mdf.csv", mdf) =#
#= global ctr = 0
for line in readlines("/home/mulle/Documents/JuliaFiles/thymus_ABM/surrogates/test.txt")
    global ctr += 1
    if ctr >= 495
        println(ctr)
        data = split(line)
        thy = trunc(Int,parse(Float64,data[1]))
        mtec = trunc(Int,parse(Float64,data[2]))
        dc = trunc(Int,parse(Float64,data[3]))
        interacts = trunc(Int,parse(Float64,data[4]))
        thresh = parse(Float64,data[5])
        model = initialize(; width_height = (10.0,10.0,10.0), n_tecs = mtec, n_dendritics = dc, 
        n_thymocytes = thy, speed = agent_speed, threshold = thresh, dt = 1.0, rng_seed = 1, treg_threshold = 0.6, 
        synapse_interactions = interacts, min_strong_interactions = 1)
        adf, mdf = run!(model, cell_move!, model_step!, 1000; adata = adata, mdata = mdata)

        open("/home/mulle/Documents/JuliaFiles/thymus_ABM/surrogates/newmyfile2.txt", "a") do io
            num1 = mdf[!, "react_ratio"][end]
            num2 = mdf[!, "nonreact_ratio"][end]
            num3 = mdf[!, "escapedautoreactive_ratio"][end]
            write(io, string(num1) * " " * string(num2) * " " * string(num3) * "\n")
        end;
    end
end =#
#= figure, adf, mdf = abm_data_exploration(
    model2, cell_move!, model_step!, parange;
    as = cell_sizes, ac = cell_colors, #adata = adata, alabels = alabels,
    mdata = mdata, mlabels = mlabels)  =#

#= abm_video(
    "thymus_abm_3Dvid_newtest.mp4",
    model2,
    cell_move!,
    model_step!;
    frames = 2000,
    ac = cell_colors,
    as = cell_sizes,
    spf = 1,
    framerate = 100,
) =#

#@benchmark run!(model2, cell_move!, model_step!, 1000; adata = adata)
#adf, mdf = run!(model2, cell_move!, model_step!, 1000; adata = adata, mdata=mdata)
#= adf, mdf = run!(model2, cell_move!, model_step!, 1000; adata = adata, mdata=mdata)
x = mdf.step
thy_data = mdf.nonautoreactive_thymocytes
figure = Figure(resolution = (600, 400))
ax = figure[1, 1] = Axis(figure, xlabel = "Steps", ylabel = "Proportion")
ax.title = "Proportion of Autoreactive to Nonautoreactive Thymocytes"
lthy = lines!(ax, x, thy_data, color = :blue)
#figure[1, 2] = Legend(figure, [lthy], ["Proportion"], textsize = 12)
display(figure) =#

#= x = mdf.step
success = mdf.successful_interactions
unsuccess = mdf.unsuccessful_interactions
figure = Figure(resolution = (600, 400))
ax = figure[1, 1] = Axis(figure, xlabel = "Steps", ylabel = "Number of Interactions")
ax.title = "Quantity of Thymocyte/APC Interactions"
lsuccess= lines!(ax, x, success, color = :blue)
lunsuccess = lines!(ax, x, unsuccess, color = :red)
figure[1, 2] = Legend(figure, [lsuccess, lunsuccess], ["Successful", "Unsuccessful"], textsize = 12)
display(figure) =#

# Runs model ensemble (in this case w/ different RNG seeds for each) and plots average thymocyte count over across all models over time
#= num_ensembles = 5
models = [initialize(; width_height = dims, n_tecs = 10, n_dendritics = 10, n_thymocytes = 1000, speed = agent_speed, threshold = 0.75, dt = 1.0, rng_seed = x, treg_threshold = 0.6) for x in rand(UInt8, num_ensembles)];
adf, mdf = ensemblerun!(models, cell_move!, model_step!, 1000; adata = adata, mdata = mdata, parallel = true)

# Make each ensemble adf data an individual element in a vector
dfs = [mdf[in([i]).(mdf.ensemble), :] for i in range(1,num_ensembles; step=1)]

# Takes mean of the ensemble-seperated vector for all data in it
dfs_mean = reduce(.+, dfs) ./ length(dfs)

# Plot relevant data
x = dfs_mean.step
thy_data = dfs_mean.react_ratio
figure = Figure(resolution = (600, 400))
ax = figure[1, 1] = Axis(figure, xlabel = "Steps", ylabel = "React Ratio")
lthy = lines!(ax, x, thy_data, color = :blue)
figure[1, 2] = Legend(figure, [lthy], ["React Ratio"], textsize = 12)
display(figure) =#
end
