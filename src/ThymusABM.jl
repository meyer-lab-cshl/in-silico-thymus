module ThymusABM

export Tec, Dendritic, Thymocyte, Parameters, add_tecs!, add_dendritics!, add_thymocytes!, calculate_reaction_strength, initialize, cell_move!, set_color!, tec_DC_interact!, thymocyte_APC_interact!, update_tec_stage, collide!, model_step!, cell_colors, cell_sizes, cell_markers, parse_commandline, tec, thymocyte, react_ratio, escape_ratio, escapedautoreactive_ratio, nonreact_ratio, total_thy, alive_ratio
#using Distributed
#using ArgParse
#include("FastPeptide.jl")
#addprocs(2; exeflags=`--project=$(Base.active_project())`)
#parsed_args = parse_commandline()
#@everywhere parsed_args = $parsed_args

#@everywhere begin
include("FastPeptide.jl")
using .FastPeptide
using Base: Float64
using Agents
using Random
using InteractiveDynamics
using Statistics: mean
using StatsBase # used to sample without replacement
using DelimitedFiles
#using NamedArrays
using NPZ
using JSON
using CSV
using ArgParse
using LinearAlgebra
using Distributions
using TOML
using CairoMakie
#using Mmap

################## optimize checking of all peptides for escaped autoreactives #################
# make Set of only genes/peptides actually present in simulation?
# or keep it to be entire .txt file? - this is how it is currently

# Integers associated with each amino acid character:
#aa2code = {"A":0, "B":21, "C":4, "D":3,
#          "E":6, "F":13, "G":7, "H":8,
#          "I":9, "J":21, "K":11, "L":10,
#          "M":12, "N":2, "O":21, "P":14,
#          "Q":5, "R":1, "S":15, "T":16,
#          "U":21, "V":19, "W":17, "X":21,
#          "Y":18, "Z":21}
# 21 should never appear - those characters do not correspond to amino acids
# Peptides are each a 1x9 Array/Matrix of Ints (Matrix == 2D Array)
# TCRs are each a 1x9 Array/Matrix of Ints
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
    antigens::Matrix{Int}            # List of antigens the Tec agent contains
    stage::Int                          # Maturation stage of Tec
    steps_alive::Int                    # Total number of steps Tec has been alive for
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
    antigens::Matrix{Int}             # List of antigens the Dendritic agent contains
    steps_alive::Int                    # Number of steps the Dendritic agent has been alive for
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
    tcr::Array{Int}                         # TCR that the thymocyte agent is carrying
    reaction_levels::Dict{Array{Int}, Int}# Dict to hold thymocyte's seen antigens and reaction levels to them
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
    deaths::Int = deaths                                                    # Int of total number of thymocytes that have exited simulation for any reason at current step
    total_dead_thymocytes::Int = total_dead_thymocytes                      # Int of total number of thymocytes that have exited simulation for any reason across all steps
    alive_thymocytes::Int = alive_thymocytes                                # Int of total number of alive thymocytes at current step
    stage_genes_peptides_dict::Vector{Any} = stage_genes_peptides_dict      # Vector holding the development stage information for mTECs (genes and corresponding peptides for each stage)
    max_tec_interactions::Int = max_tec_interactions                        # Int of maximum number of interactions a mTEC can have before it "dies" and is replaced by a new one
    max_thymocyte_interactions::Int = max_thymocyte_interactions            # Int of maximum number of interactions a thymocyte must have before it leaves as a nonautoreactive T cell
    step::Int = 0                                                           # Int of current step of simulation
    mtec_genes::Int = mtec_genes                                            # Int of number of genes mtec will express
    mtec_peptides::Int = mtec_peptides                                      # Int of number of peptides mtec will present from its genes

Parameters struct.
"""
Base.@kwdef mutable struct Parameters
    width_height::NTuple{3,Float64} = width_height                          # 3-Tuple of Float64 for 3 dimensions of the simulation (x, y, z)
    speed::Float64 = speed                                                  # Float64 speed factor
    dt::Float64 = dt                                                        # Float64 of time step governing movement/velocity of agents
    n_tecs::Int = n_tecs                                                    # Int of number of mTECs to have in simulation
    n_thymocytes::Int = n_thymocytes                                        # Int of number of thymocytes to have in simulation
    n_dendritics::Int = n_dendritics                                        # Int of number of DCs to have in simulation
    threshold::Int = threshold                                              # Int of negative selection threshold of thymocytes
    peptides::Matrix{Int} = peptides                                        # Array of all possible peptides a mTEC/DC can present                   
    successful_interactions::Int = successful_interactions                  # Int of number of times a thymocyte was selected in an interaction with mTEC/DC
    unsuccessful_interactions::Int = unsuccessful_interactions              # Int of number of times a thymocyte was not selected in an interaction with mTEC/DC
    escaped_thymocytes::Int = 0                                             # Int of number of thymocytes that were not negatively selected that should have been
    autoreactive_thymocytes::Int = 0                                        # Int of number of thymocytes that were negatively selected or escaped
    nonautoreactive_thymocytes::Int = 0                                     # Int of number of thymocytes that were not autoreactive and can be assumed to have successfully exited the thymus
    total_thymocytes::Int = 0                                               # Int of total number of thymocytes that have entered simulation
    treg_threshold::Float64 = treg_threshold                                # Float64 setting the threshold for a nonautoreactive thymocyte to be classified as a Treg
    num_tregs::Int = 0                                                      # Int of total number of Tregs that have successfully exited simulation
    tecs_present::Int = n_tecs                                              # Int of number of mTECs currently present in the simulation
    synapse_interactions::Int = synapse_interactions                        # Int of total number of peptide:TCR reactions to calculate for one thymocyte:mTEC/DC interaction
    deaths::Int = deaths                                                    # Int of total number of thymocytes that have exited simulation for any reason at current step
    total_dead_thymocytes::Int = total_dead_thymocytes                      # Int of total number of thymocytes that have exited simulation for any reason across all steps
    alive_thymocytes::Int = alive_thymocytes                                # Int of total number of alive thymocytes at current step
    stage_genes_peptides_dict::Vector{Dict{String, Array{Array{Int}}}} = stage_genes_peptides_dict      # Vector holding the development stage information for mTECs (genes and corresponding peptides for each stage)
    max_tec_interactions::Int = max_tec_interactions                        # Int of maximum number of interactions a mTEC can have before it "dies" and is replaced by a new one
    max_thymocyte_interactions::Int = max_thymocyte_interactions            # Int of maximum number of interactions a thymocyte must have before it leaves as a nonautoreactive T cell
    step::Int = 0                                                           # Int of current step of simulation
    mtec_genes::Int = mtec_genes                                            # Int of number of genes mtec will express
    mtec_peptides::Int = mtec_peptides                                      # Int of number of peptides mtec will present from its genes
    min_strength::Int = min_strength
    matrix_type::String = matrix_type
    presented_peptides::Set{Array{Int}} = presented_peptides
    expressed_genes::Set{String} = expressed_genes
    encountered_peptides_count::Vector{Tuple{Int, Bool}} = encountered_peptides_count
end

"""
    add_tecs!(model, n_tecs, color, size, replenishing)

Adds `n_tecs` number of Tecs to the `model` of given `color` and `size`. Boolean `replenishing` determines if Tec is an initial Tec (true), or added to the model later on (false).
"""
function add_tecs!(model, n_tecs::Int, color::String, size::Float64, replenishing::Bool)
    for _ in 1:n_tecs
        id = nextid(model)
        pos = Tuple(rand(model.rng, 3))
        velocity = (0.0, 0.0, 0.0)
        mass = Inf
        steps_alive = 0
        if replenishing == false
            num_interactions = rand(model.rng, 0:model.max_tec_interactions - 1)
            stage = rand(model.rng, 1:length(model.stage_genes_peptides_dict))
        else
            num_interactions = 0
            stage = 1
        end

        valid_genes_peptides = model.stage_genes_peptides_dict[stage]
        # how many genes for 1 mTEC? do we explicitly care about keeping gene names, or just combine their peptides into 1 array?
        genes = rand(model.rng, valid_genes_peptides, model.mtec_genes)
        push!(model.expressed_genes, [gene[1] for gene in genes]...)
        peptides = vcat([gene[2] for gene in genes]...) # since gene is a gene -> peptides pair, gene[2] = peptides for that gene
        peptides = sample(model.rng, peptides, model.mtec_peptides) # choose a size sample from peptides to act as tec's antigens
        peptides = Array{Array{Int64}}(peptides)#convert(Array{Array{Int64,1},1}, peptides)
        peptides = Matrix(reduce(hcat,peptides)')
        push!(model.presented_peptides, peptides)
        tec = Tec(id, pos, velocity, mass, :tec, color, size, num_interactions, peptides, stage, steps_alive)
        add_agent!(tec, model)
        set_color!(tec, model)
    end
end

"""
    add_dendritics!(model, n_dendritics, color, size)

Adds `n_dendritics` number of Dendritics to the `model` of given `color` and `size`.
"""
function add_dendritics!(model, n_dendritics::Int, color::String, size::Float64)
    # review DC peptides - how many they start with, how they gain new ones
    for _ in 1:n_dendritics
        id = nextid(model)
        pos = Tuple(rand(model.rng, 3))
        velocity = ((sincos(2π * rand(model.rng)) .* model.speed)...,sin(2π * rand(model.rng)) .* model.speed)
        mass = 1.0
        steps_alive = 0
        num_interactions = rand(model.rng, 0:model.max_tec_interactions - 1)

        stage = rand(model.rng, 1:length(model.stage_genes_peptides_dict)) # figure out best way to increment stage
        valid_genes_peptides = model.stage_genes_peptides_dict[stage]
        # how many genes for 1 mTEC? do we explicitly care about keeping gene names, or just combine their peptides into 1 array?
        genes = rand(model.rng, valid_genes_peptides, model.mtec_genes)
        peptides = vcat([gene[2] for gene in genes]...) # since gene is a gene -> peptides pair, gene[2] = peptides for that gene
        peptides = sample(model.rng, peptides, model.mtec_peptides) # choose a size sample from peptides to act as DC's antigens
        #peptides = convert(Array{Vector{Int64},1}, peptides)
        peptides = Matrix(reduce(hcat,peptides)')
        #peptides = sample(model.rng, model.peptides, 1, replace = false) # choose 1 antigen for DC to start with
        dc = Dendritic(id, pos, velocity, mass, :dendritic, color, size, num_interactions, peptides, steps_alive)
        add_agent!(dc, model)
        #set_color!(dc)
    end
end

"""
    add_thymocytes!(model, n_thymocytes, color, size, initial)

Adds `n_thymocytes` number of Thymocytes to the `model` of given `color` and `size`. Boolean `initial` determines if Thymocyte is an initial Thymocyte (true), or added to the model later on (false).
"""
function add_thymocytes!(model, n_thymocytes::Int, color::String, sizes::Float64, initial::Bool)
    for _ in 1:n_thymocytes
        id = nextid(model)
        model.total_thymocytes += 1
        model.alive_thymocytes += 1
        pos = Tuple(rand(model.rng, 3))
        vel = ((sincos(2π * rand(model.rng)) .* model.speed)...,sin(2π * rand(model.rng)) .* model.speed)
        mass = 1.0
        tcr = Matrix(rand(model.rng, 1:20, 9)')#randstring(model.rng, "ACDEFGHIKLMNPQRSTVWY", 9)
        #strengths = Array{Cint,1}(undef, size(model.peptides))
        #calc_binding_strengths(strengths, model.peptides, [tcr])
        #ccall((:calc_binding_strengths, "./data/myfplib.so"), Cvoid, (Ptr{Cint}, Ptr{UInt8}, Ptr{UInt8}), strengths, "./data/uniquepeptides.txt", tcr)
        #binding_strengths = NamedArray(strengths, (model.peptides), ("pepts"))
        #tcr = rand(model.rng, model.tcrs)
        #deleteat!(model.tcrs, findall(x->x==tcr,model.tcrs))
        steps_alive = 0
        reaction_levels = Dict{Array{Int}, Int}()
        death_label = false
        if initial == true
            finalcheck = false
            num_interactions = rand(model.rng, 0:model.max_thymocyte_interactions - 1) #if want to randomize this, also have to randomize initial reaction levels
            if num_interactions != 0
                for i in 1:num_interactions
                    rand_tec = random_agent(model, tec)
                    peptindex = rand(model.rng, axes(rand_tec.antigens, 1), 1)
                    pept = rand_tec.antigens[peptindex, :]
                    #calculate_reaction_strength(pept, tcr, reaction_levels, model.matrix_type)
                    death_label = fasthamming(pept, tcr, model.threshold, reaction_levels, model.min_strength, finalcheck)
                end
            end
        else
            num_interactions = 0
        end

        confined = false
        bind_location = (0.0,0.0,0.0)
        treg = false
        thymocyte = Thymocyte(id, pos, vel, mass, :thymocyte, color, sizes, num_interactions, tcr, reaction_levels, death_label, confined, bind_location, treg, steps_alive)
        add_agent!(thymocyte, model)
        #set_color!(thymocyte)
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
        max_tec_interactions = 200,
        max_thymocyte_interactions = 80)

Initialize the model with default or specified parameters. Returns the initialized `model`.
"""
function initialize(;
    width_height = (1.0, 1.0, 1.0)::NTuple{3,Float64},
    speed = 0.002::Float64,
    n_tecs = 50::Int,
    n_thymocytes = 1000::Int,
    n_dendritics = 50::Int,
    dt=1.0::Float64,
    threshold = 0.8::Float64,
    treg_threshold = 0.6::Float64,
    rng_seed = 1::Int,
    synapse_interactions = 1::Int,
    max_tec_interactions = 200::Int,
    max_thymocyte_interactions = 80::Int,
    mtec_genes = 100::Int,
    mtec_peptides = 100::Int,
    min_strength = 2::Int,
    matrix_type = "hamming"::String)

    rng = MersenneTwister(rng_seed)
    step = 0

    stage_genes_peptides_dict = JSON.parsefile("./data/stage_genes_peptides_intlists.json")
    stage_genes_peptides_dict = Vector{Dict{String, Array{Array{Int}}}}(stage_genes_peptides_dict)

    space3d = ContinuousSpace(width_height, width_height[1]/10) # change number here depending on volume dimensions used
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

    presented_peptides = Set{Array{Int}}()
    expressed_genes = Set{String}()
    encountered_peptides_count = Vector{Tuple{Int, Bool}}()

    #aa_data = npzread("/home/mulle/Documents/JuliaFiles/thymus_ABM/data/H2_proportional_binding_matrices.npy")
    #aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    pepts = Set{Array{Int}}()
    for i in eachindex(stage_genes_peptides_dict)
        genes = stage_genes_peptides_dict[i]
        p = vcat([gene[2] for gene in genes]...)
        for pep in p
            push!(pepts, pep)
        end
    end
    pepts = Matrix(hcat(pepts...)')
    #pepts = readlines(open("./data/uniquepeptides.txt"))
    #tcrs = readlines(open("./data/tcrs60000.txt"))
    #matches = open("./data/matchesint.dist")
    #matr = Mmap.mmap(matches, Matrix{Int16}, (size(pepts)[1], size(tcrs)[1]))
    #aa_matrix = NamedArray(matr, (pepts, tcrs), ("pepts", "TCRs"))
    #close(matches)

    properties = Parameters(width_height, speed, dt, n_tecs, n_thymocytes, n_dendritics, threshold, pepts, successful_interactions, unsuccessful_interactions, escaped_thymocytes, autoreactive_thymocytes, nonautoreactive_thymocytes,
        total_thymocytes, treg_threshold, num_tregs, n_tecs, synapse_interactions, deaths, total_dead_thymocytes, alive_thymocytes, stage_genes_peptides_dict, max_tec_interactions,
        max_thymocyte_interactions, step, mtec_genes, mtec_peptides, min_strength, matrix_type, presented_peptides, expressed_genes, encountered_peptides_count)
    
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
            push!(model.encountered_peptides_count, (length(agent.reaction_levels), agent.death_label))
            kill_agent!(agent, model)
            model.autoreactive_thymocytes += 1
            model.deaths += 1
            model.total_dead_thymocytes += 1
            return
            # Fix movement under confinement below? - some agents move back and forth over short distance - confine around location that thymocyte binded or location of binding tec or is that the same thing?
        elseif agent.confined == true
            if agent.pos[1] >= agent.bind_location[1] + model.width_height[1]/3 || agent.pos[2] >= agent.bind_location[2] + model.width_height[2]/3 || 
                agent.pos[1] >= agent.bind_location[1] - model.width_height[1]/3 || agent.pos[2] >= agent.bind_location[2] - model.width_height[2]/3 ||
                agent.pos[3] >= agent.bind_location[3] + model.width_height[3]/3 || agent.pos[3] >= agent.bind_location[3] - model.width_height[3]/3
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
    tec_DC_interact!(a1::Union{Tec, Dendritic, Thymocyte}, a2::Union{Tec, Dendritic, Thymocyte})

Governs the interaction between a Tec and Dendritic who come into contact. Transfers Tec antigens to Dendritic antigens.
"""
function tec_DC_interact!(a1::Union{Tec, Dendritic, Thymocyte}, a2::Union{Tec, Dendritic, Thymocyte})
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
        if thymocyte_agent.num_interactions >= model.max_thymocyte_interactions
            return
        end
    elseif a1.type == :thymocyte && (a2.type == :tec || a2.type == :dendritic)
        tec_agent = a2
        thymocyte_agent = a1
        if thymocyte_agent.num_interactions >= model.max_thymocyte_interactions
            return
        end
    else
        return
    end
    # compare a chosen tec antigen sequence to thymocyte TCR sequence
    # choose random antigen from tec's antigens to compare thymocyte tcr to. use aa_matrix to retrieve stength of interaction, comparing characters one by one
    # if a reaction level passes the model threshold, the thymocyte is killed
    #antigens = rand(model.rng, tec_agent.antigens, model.synapse_interactions)
    peptindex = rand(model.rng, axes(tec_agent.antigens, 1), model.synapse_interactions)
    antigens = tec_agent.antigens[peptindex, :]

    finalcheck = false
    thymocyte_agent.death_label = fasthamming(antigens, thymocyte_agent.tcr, model.threshold, thymocyte_agent.reaction_levels, model.min_strength, finalcheck)
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
            genes = rand(model.rng, valid_genes_peptides, model.mtec_genes)
            push!(model.expressed_genes, [gene[1] for gene in genes]...)
            pepts = vcat([gene[2] for gene in genes]...) # since gene is a gene -> peptides pair, gene[2] = peptides for that gene
            pepts = sample(model.rng, pepts, model.mtec_peptides) # choose a size sample from peptides to act as tec's antigens
            pepts = convert(Array{Vector{Int64},1}, pepts)
            tec.antigens = Matrix(reduce(hcat,pepts)')
            push!(model.presented_peptides, tec.antigens)
            tec.num_interactions = 0
        end
    end
end

"""
    collide!(a::AbstractAgent, b::AbstractAgent)

Calculates an elastic collision between two agents, `a` and `b`, when they collide with each other. Adapted from Agents.jl elastic_collision and modified to work for 3D animations.
"""
function collide!(a::AbstractAgent, b::AbstractAgent)
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
    for (a1, a2) in interacting_pairs(model, interaction_radius, :types)
        #:types allows for easy changing of tec size by changing radius, but then thymocytes do not interact at all. :all causes error if thymocyte is deleted while having > 1 interaction. :nearest only allows for 1 interaction 
        if (a1.type == :tec || a1.type == :dendritic) && (a2.type == :tec || a2.type == :dendritic)
            tec_DC_interact!(a1, a2)
        elseif !(a1.type == :thymocyte && a2.type == :thymocyte)
            thymocyte_APC_interact!(a1, a2, model)
        end
        collide!(a1, a2)
    end
    for agent in allagents(model)
        agent.steps_alive += 1
        escaped = false
        if (agent.type == :tec && agent.stage >= length(model.stage_genes_peptides_dict) + 1) || (agent.num_interactions >= model.max_thymocyte_interactions && agent.type == :thymocyte)
            if agent.type == :tec
                model.tecs_present -= 1 
            elseif agent.type == :thymocyte
                push!(model.encountered_peptides_count, (length(agent.reaction_levels), agent.death_label))
                model.deaths += 1
                model.total_dead_thymocytes += 1
                escaped = fasthamming(model.peptides, agent.tcr, model.threshold, agent.reaction_levels, model.min_strength, true)

                if agent.treg == true
                    model.num_tregs += 1
                end
                if escaped == false
                    model.nonautoreactive_thymocytes += 1
                else
                    model.autoreactive_thymocytes += 1
                    model.escaped_thymocytes += 1
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
    tec(a)

Used to check if agent `a` is a Tec.
"""
tec(a) = a.type == :tec

"""
    thymocyte(a)

Used to check if agent `a` is a Thymocyte.
"""
thymocyte(a) = a.type == :thymocyte

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

"""
    model_volume(model)

Return the volume of the `model`.
"""
model_volume(model) = model.width_height[1] * model.width_height[2] * model.width_height[3]

"""
    agent_peptides_encountered(agent)

Return the number of peptides that `agent` has encountered.
"""
agent_peptides_encountered(agent) = length(agent.reaction_levels)

"""
    read_parameters(filename::String)

Obtain and return the model parameters from the config file named `filename`.
"""
function read_parameters(filename::String)
    params = TOML.parsefile(filename)
    rng = params["rng"]
    speed = params["thymocyte_speed"]
    selection_threshold = params["selection_threshold"]
    treg_threshold = params["treg_threshold"]
    synapse_interactions = params["synapse_interactions"]
    volume = params["volume"]#rand(MersenneTwister(rng), dist, num_ensembles)
    steps = params["steps"]
    num_medullas = params["num_medullas"]
    min_strength = params["minimum_strength"]
    matrixtype = params["matrixtype"]
    return rng, speed, selection_threshold, treg_threshold, synapse_interactions, volume, steps, num_medullas, min_strength, matrixtype
end

check_stage_1(agent) = agent.stage == 1
check_stage_2(agent) = agent.stage == 2
check_stage_3(agent) = agent.stage == 3
check_stage_4(agent) = agent.stage == 4
check_stage_5(agent) = agent.stage == 5
check_stage_6(agent) = agent.stage == 6
check_stage_7(agent) = agent.stage == 7
check_stage_8(agent) = agent.stage == 8
check_stage_9(agent) = agent.stage == 9

"""
    run()

Run the model.
"""
function run()
    adata = [(tec, count), (check_stage_1, count, tec), (check_stage_2, count, tec), (check_stage_3, count, tec), (check_stage_4, count, tec), (check_stage_5, count, tec), (check_stage_6, count, tec), (check_stage_7, count, tec), (check_stage_8, count, tec), (check_stage_9, count, tec),]
    mdata = [:num_tregs, :autoreactive_thymocytes, :escaped_thymocytes, :nonautoreactive_thymocytes, :alive_thymocytes, escape_ratio, react_ratio, nonreact_ratio, :threshold, total_thy, alive_ratio, escapedautoreactive_ratio, model_volume]
    mlabels = ["number of tregs", "autoreactive", "escaped", "nonautoreactive", "alive", "escaped thymocytes ratio", "autoreactive thymocytes ratio", "nonautoreactive ratio", "selection threshold", "total thymocytes", "alive thymocytes ratio", "escaped to autoreactive ratio", "volume"]
    
    #model2 = initialize(; width_height = tuple(parsed_args["dim1"], parsed_args["dim2"], parsed_args["dim3"]), n_tecs = parsed_args["n_tecs"], n_dendritics = parsed_args["n_dendritics"], 
    #n_thymocytes = parsed_args["n_thymocytes"], speed = parsed_args["thymocyte_speed"], threshold = parsed_args["selection_threshold"], dt = parsed_args["dt"], rng_seed = parsed_args["rng"], treg_threshold = parsed_args["treg_threshold"], 
    #synapse_interactions = parsed_args["synapse_interactions"])

    rng, speed, selection_threshold, treg_threshold, synapse_interactions, volume, steps, num_medullas, min_strength, matrixtype = read_parameters("./data/config.toml")

    if volume == "RANDOM"
        GM = 0.00016
        GSD = 9.97
        volumemean = log(GM)
        volumestd = log(GSD)
        distrib = LogNormal(volumemean, volumestd)
        generator = MersenneTwister(rng)
        volumes = rand(generator, distrib, num_medullas)
    else
        volumes = [volume for i in 1:num_medullas]
    end
    
    dimensions = Array{NTuple{3, Float64}, 1}(undef, num_medullas)
    thymocytes = Array{Int, 1}(undef, num_medullas)
    tecs = Array{Int, 1}(undef, num_medullas)
    dcs = Array{Int, 1}(undef, num_medullas)
    for i in eachindex(volumes)
        dim = cbrt(volumes[i])
        dimensions[i] = (dim, dim, dim)
        # 12500000 thymocytes in negative selection. Largest medulla of 2.0 mm^3 is 73.6% of total medulla volume. .736 * 12500000 = 9200000 thymocytes in largest medulla. 9200000 / 2 = 4600000 per mm^3
        thymocytes[i] = Integer(trunc(4600000 * volumes[i]))
        if thymocytes[i] < 1
            thymocytes[i] = 1
        end
        tecs[i] = Integer(trunc(956800 * volumes[i])) # 2600000 mTECs in negative selection. (2600000 * .736) / 2 = 956800 per mm^3
        if tecs[i] < 1
            tecs[i] = 1
        end
        dcs[i] = Integer(trunc(95680 * volumes[i])) # take to be 1/10th of mTEC value
        if dcs[i] < 1
            dcs[i] = 1
        end
    end
    models = [initialize(; width_height = dimensions[i], n_tecs = tecs[i], n_dendritics = dcs[i], n_thymocytes = thymocytes[i], speed = speed, 
    threshold = selection_threshold, dt = 1.0, rng_seed = rng[i], treg_threshold = treg_threshold, synapse_interactions = synapse_interactions, 
    min_strength = min_strength, matrix_type = matrixtype) for i in 1:num_medullas];
    #adf, mdf = ensemblerun!(models, cell_move!, model_step!, 1000; adata = adata, mdata = mdata, parallel = true)
    @time Threads.@threads for i = 1:num_medullas
        adf, mdf = run!(models[i], cell_move!, model_step!, steps; adata = adata, mdata = mdata)
        adfname = "adf" * string(thymocytes[i]) * "thymocytes" * string(selection_threshold) * "threshold" * string(synapse_interactions) * "synapsecomplexes" * string(rng[i]) * "rngseed.csv"
        mdfname = "mdf" * string(thymocytes[i]) * "thymocytes" * string(selection_threshold) * "threshold" * string(synapse_interactions) * "synapsecomplexes" * string(rng[i]) * "rngseed.csv"
        CSV.write("./data/results/" * adfname, adf)
        CSV.write("./data/results/" * mdfname, mdf)
        println("Simulation complete. Data written to ThymusABM/data folder as " * adfname * " and " * mdfname)
        #writedlm( "./data/newhamtestpeptidecountdata.csv",  models[i].encountered_peptides_count, ',')
        #writedlm( "./data/synapse/synapse2/25synapsegenes003volume.csv",  models[i].expressed_genes, ',')
    end


    #global parange = Dict(:threshold => 0:0.01:1)
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
end
#run()


end