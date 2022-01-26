#using Distributed
#addprocs(2)
#@everywhere begin
using Base: Float64
using Agents: length, isempty, getindex
using Agents
using Random
using InteractiveDynamics
using GLMakie # GLMakie needed for abm_data_exploration, CairoMakie does not work correctly with it but does work for plots/videos
using Statistics: mean
using StatsBase # used to sample without replacement
using DelimitedFiles
using NamedArrays
using NPZ
using JSON
using CSV
using ArgParse

################## optimize checking of all peptides for escaped autoreactives #################
# make Set of only genes/peptides actually present in simulation?
# or keep it to be entire .txt file? - this is how it is currently
################## check tec_DC interaction ####################################################
# how to transfer peptides?
# all or only some? - transfer all for now

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
end

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
end

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
end

Base.@kwdef mutable struct Parameters
    width_height::NTuple{3,Float64} = width_height
    speed::Float64 = speed
    dt::Float64 = dt
    n_tecs::Int = n_tecs
    n_thymocytes::Int = n_thymocytes
    n_dendritics::Int = n_dendritics
    threshold::Float64 = threshold
    peptides::Array = peptides
    successful_interactions::Int = successful_interactions
    unsuccessful_interactions::Int = unsuccessful_interactions
    escaped_thymocytes::Int = 0
    autoreactive_thymocytes::Int = 0
    nonautoreactive_thymocytes::Int = 0
    total_thymocytes::Int = 0
    treg_threshold::Float64 = treg_threshold
    num_tregs::Int = 0
    tecs_present::Int = n_tecs
    aa_matrix::NamedArray = aa_matrix
    synapse_interactions::Int = synapse_interactions
    min_strong_interactions::Int = min_strong_interactions
    total_peptides::Int = total_peptides
    deaths::Int = deaths
    total_dead_thymocytes::Int = total_dead_thymocytes
    alive_thymocytes::Int = alive_thymocytes
    stage_genes_peptides_dict::Vector{Any} = stage_genes_peptides_dict
    max_tec_interactions::Int = max_tec_interactions
    max_thymocyte_interactions::Int = max_thymocyte_interactions
end

function add_tecs!(model, n_tecs, color, size, replenishing)
    for _ in 1:n_tecs
        id = nextid(model)
        pos = Tuple(rand(model.rng, 3))
        velocity = (0.0, 0.0, 0.0)
        mass = Inf
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
        tec = Tec(id, pos, velocity, mass, :tec, color, size, num_interactions, peptides, stage)
        add_agent!(tec, model)
        set_color!(tec, model)
    end
end

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
        num_interactions = rand(model.rng, 0:model.max_tec_interactions - 1)

        stage = rand(model.rng, 1:length(model.stage_genes_peptides_dict)) # figure out best way to increment stage
        valid_genes_peptides = model.stage_genes_peptides_dict[stage]
        # how many genes for 1 mTEC? do we explicitly care about keeping gene names, or just combine their peptides into 1 array?
        # check rng
        genes = rand(model.rng, valid_genes_peptides, 100)
        peptides = vcat([gene[2] for gene in genes]...) # since gene is a gene -> peptides pair, gene[2] = peptides for that gene

        #antis = sample(model.rng, model.peptides, 1, replace = false) # choose 1 antigen for DC to start with
        dc = Dendritic(id, pos, velocity, mass, :dendritic, color, size, num_interactions, peptides)
        add_agent!(dc, model)
        #set_color!(dc)
    end
end

function add_thymocytes!(model, n_thymocytes, color, size, initial)
    for _ in 1:n_thymocytes
        id = nextid(model)
        model.total_thymocytes += 1
        model.alive_thymocytes += 1
        pos = Tuple(rand(model.rng, 3))
        vel = ((sincos(2π * rand(model.rng)) .* model.speed)...,sin(2π * rand(model.rng)) .* model.speed)
        mass = 1.0
        tcr = randstring(model.rng, "ACDEFGHIKLMNPQRSTVWY", 9)

        if initial == true
            num_interactions = rand(model.rng, 0:model.max_thymocyte_interactions - 1) #if want to randomize this, also have to randomize initial reaction levels
            reaction_levels = Dict{String, Float64}()
            if num_interactions != 0
                for i in 1:num_interactions
                    pept = rand(model.rng, model.peptides)
                    calculate_reaction_strength(model, pept, tcr, reaction_levels)
                end
            end
        else
            num_interactions = 0
            reaction_levels = Dict{String, Float64}()
        end

        death_label = false
        confined = false
        bind_location = (0.0,0.0,0.0)
        treg = false
        thymocyte = Thymocyte(id, pos, vel, mass, :thymocyte, color, size, num_interactions, tcr, reaction_levels, death_label, confined, bind_location, treg)
        add_agent!(thymocyte, model)
        #set_color!(thymocyte)
    end
end

function calculate_reaction_strength(model, peptide, tcr, reaction_levels)
    reaction = 1.0
    for i in range(1, length(peptide), step=1)
        antigen_aa = peptide[i]
        tcr_aa = tcr[i]
        reaction *= model.aa_matrix[i, tcr_aa, antigen_aa]
    end
    reaction = reaction^(1/length(peptide))

    if get(reaction_levels, peptide, 0) != 0 # if thymocyte has seen antigen before, add to its current reaction level
        reaction_levels[peptide] += reaction
    else # otherwise, add antigen as a new entry to the reaction_levels dict
        reaction_levels[peptide] = reaction
    end
end

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
    total_peptides = 10000,
    max_tec_interactions = 200,
    max_thymocyte_interactions = 80)

    rng = MersenneTwister(rng_seed)

    possible_antigens = readdlm("./data/validpeptides.txt",'\n')
    #peptides = sample(rng, possible_antigens, total_peptides, replace=false)
    peptides = unique(vec(possible_antigens))

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
    # Data from Derivation of an amino acid similarity matrix for peptide:MHC binding and its application as a Bayesian prior
    #aa_data, header = readdlm("/home/mulle/Downloads/12859_2009_3124_MOESM2_ESM.MAT", header=true)
    #aa_matrix = NamedArray(aa_data, (vec(header), vec(header)), ("Rows", "Cols"))
    aa_data = npzread("./data/H2_proportional_binding_matrices.npy")
    aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    aa_matrix = NamedArray(aa_data,(collect(1:9), aas, aas), ("Pos","Rows","Cols"))

    properties = Parameters(width_height, speed, dt, n_tecs, n_thymocytes, n_dendritics, threshold, peptides, successful_interactions, unsuccessful_interactions, escaped_thymocytes, autoreactive_thymocytes, nonautoreactive_thymocytes,
        total_thymocytes, treg_threshold, num_tregs, n_tecs, aa_matrix, synapse_interactions, min_strong_interactions, total_peptides, deaths, total_dead_thymocytes, alive_thymocytes, stage_genes_peptides_dict, max_tec_interactions,
        max_thymocyte_interactions)
    
    model = ABM(Union{Tec, Dendritic, Thymocyte}, space3d; properties, rng,)

    # Add agents to the model
    add_tecs!(model, n_tecs, "#00ffff", 0.5, false)
    add_dendritics!(model, n_dendritics, "#ffa500", 0.5)
    add_thymocytes!(model, n_thymocytes, "#edf8e9", 0.2, true)

    return model
end

## Agent steps
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
    
    strong_reactions = 0
    for antigen in antigens
        calculate_reaction_strength(model, antigen, thymocyte_agent.tcr, thymocyte_agent.reaction_levels)
        #thymocyte_agent.reaction_levels[antigen] = min(thymocyte_agent.reaction_levels[antigen], 1.0) # clip reaction strength to max of 1.0
        
        if thymocyte_agent.reaction_levels[antigen] >= model.threshold
            strong_reactions += 1
        else
            if thymocyte_agent.reaction_levels[antigen] >= model.treg_threshold && thymocyte_agent.reaction_levels[antigen] < model.threshold
                thymocyte_agent.treg = true
            end
            model.unsuccessful_interactions += 1
        end

        if strong_reactions >= model.min_strong_interactions # kill thymocyte if sequence matches are above model threshold - (> or >=?)
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

using LinearAlgebra
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

function model_step!(model) # happens after every agent has acted
    interaction_radius = 0.06*model.width_height[1]
    for (a1, a2) in interacting_pairs(model, interaction_radius, :types) # check :all versus :nearest versus :types.
        #:types allows for easy changing of tec size by changing radius, but then thymocytes do not interact at all. :all causes error if thymocyte is deleted while having > 1 interaction. :nearest only allows for 1 interaction 
        tec_DC_interact!(a1, a2, model)
        thymocyte_APC_interact!(a1, a2, model)
        #elastic_collision!(a1, a2, :mass)
        collide!(a1, a2)
    end

    for agent in allagents(model)
        escaped = false
        if (agent.type == :tec && agent.stage >= length(model.stage_genes_peptides_dict) + 1) || (agent.num_interactions >= model.max_thymocyte_interactions && agent.type == :thymocyte)
            if agent.type == :tec
                model.tecs_present -= 1 
            elseif agent.type == :thymocyte
                model.deaths += 1
                model.total_dead_thymocytes += 1
                strong_reactions = 0
                for antigen in model.peptides # check if exiting thymocyte was autoreactive by comparing its TCR to every peptide possible to be presented
                    reaction = 1.0
                    for i in range(1, length(antigen), step=1)
                        antigen_aa = antigen[i]
                        tcr_aa = agent.tcr[i]
                        reaction *= model.aa_matrix[i, tcr_aa, antigen_aa]
                    end
                    reaction = reaction^(1/length(antigen))
            
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

cell_colors(a) = a.color
cell_sizes(a) = a.size
function cell_markers(a::Union{Tec, Dendritic, Thymocyte})
    if a.type == :thymocyte
        return :circle
    elseif a.type == :tec
        return :star5
    else
        return :diamond
    end
end

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
            default = 1.6
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

tec(a) = a.type == :tec
thymocyte(a) = a.type == :thymocyte

global adata = [(thymocyte, count), (tec, count)]
global alabels = ["Alive Thymocytes", "Alive mTECs"]

react_ratio(model) = model.autoreactive_thymocytes/model.total_thymocytes # proportion of total killed/exited thymocytes that were autoreactive
escape_ratio(model) = model.escaped_thymocytes/model.total_thymocytes  # proportion of total killed/exited thymocytes that escaped
escapedautoreactive_ratio(model) = model.escaped_thymocytes/model.autoreactive_thymocytes # proportion of total autoreactive thymocytes that escaped
nonreact_ratio(model) = model.nonautoreactive_thymocytes/model.total_thymocytes  # proportion of total killed/exited thymocytes that were nonautoreactive
total_thy(model) = model.total_thymocytes
alive_ratio(model) = model.alive_thymocytes/model.total_thymocytes


global mdata = [:num_tregs, :autoreactive_thymocytes, :escaped_thymocytes, :nonautoreactive_thymocytes, :alive_thymocytes, escape_ratio, react_ratio, nonreact_ratio, :threshold, total_thy, alive_ratio, escapedautoreactive_ratio]
global mlabels = ["number of tregs", "autoreactive", "escaped", "nonautoreactive", "alive", "escaped thymocytes ratio", "autoreactive thymocytes ratio", "nonautoreactive ratio", "selection threshold", "total thymocytes", "alive thymocytes ratio", "escaped to autoreactive ratio"]
#mdata = [:autoreactive_thymocytes, :nonautoreactive_thymocytes, :escaped_thymocytes]
#mlabels = ["Autoreactive Thymocytes", "Non-autoreactive Thymocytes", "Escaped Autoreactive Thymocytes"]

global dims = (10.0, 10.0, 10.0) # seems to work best for 3D
global agent_speed = 0.0015 * dims[1]
#model2 = initialize(; width_height = dims, n_tecs = 500, n_dendritics = 50, n_thymocytes = 5000, speed = agent_speed, threshold = 1.6, dt = 1.0, rng_seed = 42, treg_threshold = 0.6, synapse_interactions = 3, min_strong_interactions = 1,
#    total_peptides = 1000)
parsed_args = parse_commandline()
model2 = initialize(; width_height = tuple(parsed_args["dim1"], parsed_args["dim2"], parsed_args["dim3"]), n_tecs = parsed_args["n_tecs"], n_dendritics = parsed_args["n_dendritics"], 
n_thymocytes = parsed_args["n_thymocytes"], speed = agent_speed, threshold = parsed_args["selection_threshold"], dt = parsed_args["dt"], rng_seed = parsed_args["rng"], treg_threshold = 0.6, 
synapse_interactions = parsed_args["synapse_interactions"], min_strong_interactions = parsed_args["min_strong_interactions"], total_peptides = 1000)

global parange = Dict(:threshold => 0:0.01:1)

#@time adf, mdf = run!(model2, cell_move!, model_step!, 5000; adata = adata, mdata = mdata)
@time adf, mdf = run!(model2, cell_move!, model_step!, parsed_args["steps"]; adata = adata, mdata = mdata)

CSV.write("/home/mulle/Documents/JuliaFiles/thymus_ABM/abm_results/Dec15/adfnewtest.csv", adf)
CSV.write("/home/mulle/Documents/JuliaFiles/thymus_ABM/abm_results/Dec15/mdfnewtest.csv", mdf)
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