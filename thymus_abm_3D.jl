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
using BenchmarkTools
using DelimitedFiles
using NamedArrays
using NPZ

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
    age::Int                            # Age of agent; incremented after n steps
    just_aged::Bool                     # Boolean to determine if agent just aged (true) on current model step
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
    age::Int                            # Age of agent; incremented after n steps
    just_aged::Bool                     # Boolean to determine if agent just aged (true) on current model step
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
    age::Int                            # Age of agent; incremented after n steps
    just_aged::Bool                     # Boolean to determine if agent just aged (true) on current model step
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
    total_peptides = 10000)

    rng = MersenneTwister(rng_seed)

    possible_antigens = readdlm("/home/mulle/Documents/JuliaFiles/thymus_ABM/validpeptides.txt",'\n')
    peptides = sample(rng, possible_antigens, total_peptides, replace=false)

    #possible_antigens = [randstring(rng, "ACDEFGHIKLMNPQRSTVWY", 9) for i=1:45] # represents the 20 amino acids

    space3d = ContinuousSpace(width_height, 1.0) # change number here depending on volume dimensions used

    successful_interactions = 0
    unsuccessful_interactions = 0
    autoreactive_thymocytes = 0
    nonautoreactive_thymocytes = 0
    escaped_thymocytes = 0
    deaths = 0
    total_dead_thymocytes = 0
    alive_thymocytes = 10000

    total_thymocytes = 0
    num_tregs = 0
    # Data from Derivation of an amino acid similarity matrix for peptide:MHC binding and its application as a Bayesian prior
    #aa_data, header = readdlm("/home/mulle/Downloads/12859_2009_3124_MOESM2_ESM.MAT", header=true)
    #aa_matrix = NamedArray(aa_data, (vec(header), vec(header)), ("Rows", "Cols"))
    aa_data = npzread("/home/mulle/Documents/JuliaFiles/thymus_ABM/binding_matrices/H2_proportional_binding_matrices.npy")
    aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    aa_matrix = NamedArray(aa_data,(collect(1:9), aas, aas), ("Pos","Rows","Cols"))

    properties = Parameters(width_height, speed, dt, n_tecs, n_thymocytes, n_dendritics, threshold, peptides, successful_interactions, unsuccessful_interactions, escaped_thymocytes, autoreactive_thymocytes, nonautoreactive_thymocytes,
     total_thymocytes, treg_threshold, num_tregs, n_tecs, aa_matrix, synapse_interactions, min_strong_interactions, total_peptides, deaths, total_dead_thymocytes, alive_thymocytes)
    
    model = ABM(Union{Tec, Dendritic, Thymocyte}, space3d; properties, rng,)

    # Add agents to the model
    id = 0
    for _ in 1:n_tecs
        id += 1
        pos = Tuple(rand(model.rng, 3))
        if !isempty(nearby_ids(pos, model, model.width_height[1]/2)) # make sure mTECs don't overlap
            pos = Tuple(rand(model.rng, 3))
        end
        vel = (0.0, 0.0, 0.0)
        mass = Inf
        color = "#00ffff"
        size = 0.5
        num_interactions = 0
        age = rand(model.rng, 0:19) # randomize starting ages
        just_aged = false
        antis = sample(model.rng, peptides, age+100, replace = false) # choose an (age+2000) size sample from peptides to act as tec's antigens (replace = false to avoid repetitions, or are repetitions wanted?)
        tec = Tec(id, pos, vel, mass, :tec, color, size, num_interactions, antis, age, just_aged)
        add_agent!(tec, model)
        set_color!(tec)
    end

    for _ in 1:n_dendritics
        id += 1
        pos = Tuple(rand(model.rng, 3))
        if !isempty(nearby_ids(pos, model, model.width_height[1]/2)) # make sure mTECs/dendritics don't overlap
            pos = Tuple(rand(model.rng, 3))
        end
        vel = ((sincos(2π * rand(model.rng)) .* model.speed)...,sin(2π * rand(model.rng)) .* model.speed)
        mass = 1.0
        color = "#ffa500"
        size = 0.5
        num_interactions = 0
        age = rand(model.rng, 0:19) # randomize starting ages
        just_aged = false
        antis = sample(model.rng, peptides, 1, replace = false) # choose 1 antigen for DC to start with
        dc = Dendritic(id, pos, vel, mass, :dendritic, color, size, num_interactions, antis, age, just_aged)
        add_agent!(dc, model)
        #set_color!(dc)
    end

    for _ in 1:n_thymocytes
        id += 1
        model.total_thymocytes += 1
        pos = Tuple(rand(model.rng, 3))
        vel = ((sincos(2π * rand(model.rng)) .* model.speed)...,sin(2π * rand(model.rng)) .* model.speed)
        mass = 1.0
        color = "#006400"
        size = 0.2
        num_interactions = 0
        age = rand(model.rng, 0:3) # randomize starting ages
        just_aged = false
        reaction_levels = Dict{String, Float64}()

        tcr = randstring(model.rng, "ACDEFGHIKLMNPQRSTVWY", 9)

        death_label = false
        confined = false
        bind_location = (0.0,0.0,0.0)
        treg = false
        thymocyte = Thymocyte(id, pos, vel, mass, :thymocyte, color, size, num_interactions, tcr, age, just_aged, reaction_levels, death_label, confined, bind_location, treg)
        add_agent!(thymocyte, model)
        set_color!(thymocyte)
    end
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
            move_agent!(agent, model, model.dt)
        else
            move_agent!(agent, model, model.dt)
        end
        # add randomness to thymocytes' movements so they don't continue in same direction forever - maybe too much? fixes confinement movement though, but increases their velocity
        #walk!(agent, (rand(model.rng, -model.width_height[1]:model.width_height[2]) .* model.speed,rand(model.rng, -model.width_height[1]:model.width_height[2]) .* model.speed), model)
    elseif agent.type == :dendritic
        move_agent!(agent, model, model.dt)
    end
    set_color!(agent)
end

function set_color!(agent::Union{Tec, Dendritic, Thymocyte})
    if agent.type == :tec
        if agent.age > 0
            if agent.age <= 4
                agent.color = "#00ddff"
            elseif agent.age <= 8
                agent.color = "#00aaff"
            elseif agent.age <= 12
                agent.color = "#0011ff"
            elseif agent.age <= 16
                agent.color = "#0044ff"
            elseif agent.age <= 20
                agent.color = "#000000"
            end
        end
    elseif agent.type == :thymocyte
        if agent.confined == true
            agent.color = "#ffff00"
        else
            agent.color = "#ff0000"
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

    peptides = rand(model.rng, tec_agent.antigens, model.synapse_interactions)

    push!(dendritic_agent.antigens, peptides...)#rand(model.rng, model.peptides))
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

    thymocyte_agent.num_interactions += 1
    tec_agent.num_interactions += 1 

    # compare a chosen tec antigen sequence to thymocyte TCR sequence
    # choose random antigen from tec's antigens to compare thymocyte tcr to. use aa_matrix to retrieve stength of interaction, comparing characters one by one
    # if a reaction level passes the model threshold, the thymocyte is killed
    antigens = rand(model.rng, tec_agent.antigens, model.synapse_interactions)
    
    strong_reactions = 0
    for antigen in antigens
        # reaction strength is geometric mean
        reaction = 1.0
        for i in range(1, length(antigen), step=1)
            antigen_aa = antigen[i]
            tcr_aa = thymocyte_agent.tcr[i]
            reaction *= model.aa_matrix[i, tcr_aa, antigen_aa]
        end
        reaction = reaction^(1/length(antigen))

        if get(thymocyte_agent.reaction_levels, antigen, 0) != 0 # if thymocyte has seen antigen before, add to its current reaction level
            thymocyte_agent.reaction_levels[antigen] += reaction
        else # otherwise, add antigen as a new entry to the reaction_levels dict
            thymocyte_agent.reaction_levels[antigen] = reaction
        end
        
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

    thresh1 = a1.type == :thymocyte ? 10 : 20
    thresh2 = a2.type == :thymocyte ? 10 : 20
    if a1.num_interactions != 0 && a1.num_interactions % thresh1 == 0 # increment age after every thresh of agent's interactions
        a1.age += 1
        a1.just_aged = true
        set_color!(a1)
    end
    if a2.num_interactions != 0 && a2.num_interactions % thresh2 == 0
        a2.age += 1
        a2.just_aged = true
        set_color!(a2)
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

    if model.tecs_present < model.n_tecs # generate new tec if one dies
        model.tecs_present += 1
        #overlap = true
        #= while overlap == true =# # should be good to make sure new tecs don't overlap an existing tec - could be infinite loop or very long as less positions are available - though should never infinite loop if always replacing a dead one
            pos = Tuple(rand(model.rng, 3))
#=             for a in nearby_ids(pos, model, model.width_height[1]/2)
                if getindex(model, a).type == :tec || getindex(model, a).type == :dendritic
                    break
                end
                overlap = false
            end
        end =#
        vel = (0.0, 0.0, 0.0)
        antis = sample(model.rng, model.peptides, 1000, replace = false) # choose a sample from peptides to act as tec's antigens (replace = false to avoid repetitions, or are repetitions wanted?)
        tec = Tec(nextid(model), pos, vel, Inf, :tec, "#00c8ff", 0.5, 0, antis, 0, false)
        add_agent!(tec, model)
    end

    for agent in allagents(model) # kill agent if it reaches certain age and update model properties depending on agent type/properties
        escaped = false
        if (agent.age >= 20 && agent.type == :tec) || (agent.age >= 4 && agent.type == :thymocyte)
            if agent.type == :tec
                model.tecs_present -= 1 
            end
            if agent.type == :thymocyte
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

        if agent.type == :tec && agent.age != 0 && agent.just_aged == true # add new antigen to list of tec antigens as it ages
            push!(agent.antigens, sample(model.rng, model.peptides, 5, replace = false)...)#rand(model.rng, model.peptides))
            agent.just_aged = false
        end
    end

    model.alive_thymocytes = count(i->(i.type == :thymocyte), allagents(model))

    for _ in 1:model.deaths # random chance to generate new thymocyte
        pos = Tuple(rand(model.rng, 3))
        vel = ((sincos(2π * rand(model.rng)) .* model.speed)...,sin(2π * rand(model.rng)) .* model.speed)
        tcr = randstring(model.rng, "ACDEFGHIKLMNPQRSTVWY", 9)

        thymocyte = Thymocyte(nextid(model), pos, vel, 1.0, :thymocyte, "#006400", 0.2, 0, tcr, 0, false, Dict(), false, false, (0.0,0.0,0.0), false)
        add_agent!(thymocyte, model)
        model.total_thymocytes += 1
    end
    model.deaths = 0
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
tec(a) = a.type == :tec
thymocyte(a) = a.type == :thymocyte

adata = [(thymocyte, count)]
alabels = ["thymocyte count"]
#adata = [(thymocyte, count)]
#alabels = ["thymocyte count"]

react_ratio(model) = model.autoreactive_thymocytes/model.total_dead_thymocytes # proportion of total killed/exited thymocytes that were autoreactive
escape_ratio(model) = model.escaped_thymocytes/model.total_dead_thymocytes  # proportion of total killed/exited thymocytes that escaped
escapedautoreactive_ratio(model) = model.escaped_thymocytes/model.autoreactive_thymocytes # proportion of total autoreactive thymocytes that escaped
nonreact_ratio(model) = model.nonautoreactive_thymocytes/model.total_dead_thymocytes  # proportion of total killed/exited thymocytes that were nonautoreactive
total_thy(model) = model.total_thymocytes
alive_ratio(model) = model.alive_thymocytes/model.total_thymocytes

mdata = [:num_tregs, :autoreactive_thymocytes, :escaped_thymocytes, :nonautoreactive_thymocytes, :alive_thymocytes, escape_ratio, react_ratio, nonreact_ratio, :threshold, total_thy, alive_ratio, escapedautoreactive_ratio]
mlabels = ["number of tregs", "autoreactive", "escaped", "nonautoreactive", "alive", "escaped thymocytes ratio", "autoreactive thymocytes ratio", "nonautoreactive ratio", "selection threshold", "total thymocytes", "alive thymocytes ratio", "escaped to autoreactive ratio"]

dims = (10.0, 10.0, 10.0) # seems to work best for 3D
agent_speed = 0.0015 * dims[1]
model2 = initialize(; width_height = dims, n_tecs = 100, n_dendritics = 10, n_thymocytes = 1000, speed = agent_speed, threshold = 1.0, dt = 1.0, rng_seed = 42, treg_threshold = 0.6, synapse_interactions = 1, min_strong_interactions = 1,
    total_peptides = 1000)

parange = Dict(:threshold => 0:0.01:1)

#= figure, adf, mdf = abm_data_exploration(
    model2, cell_move!, model_step!, parange;
    as = cell_sizes, ac = cell_colors, adata = adata, alabels = alabels,
    mdata = mdata, mlabels = mlabels)  =#

abm_video(
    "thymus_abm_3Dvid_newtest.mp4",
    model2,
    cell_move!,
    model_step!;
    frames = 2000,
    ac = cell_colors,
    as = cell_sizes,
    spf = 1,
    framerate = 100,
)

#@benchmark run!(model2, cell_move!, model_step!, 1000; adata = adata)
#adf, mdf = run!(model2, cell_move!, model_step!, 1000; adata = adata, mdata=mdata)
#= adf, mdf = run!(model2, cell_move!, model_step!, 1000; adata = adata, mdata=mdata)
x = mdf.step
thy_data = mdf.react_ratio
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
