using Base: Float64
using Agents: length
using Agents
using Random
using InteractiveDynamics
using GLMakie # Needed for abm_data_exploration, CairoMakie does not work correctly with it but does work for plots/videos
using DrWatson: @dict
using Statistics: mean
using StatsBase # used to sample without replacement

mutable struct Tec <: AbstractAgent
    id::Int                             # Unique ID to identify agent
    pos::NTuple{2,Float64}              # 2D position of agent
    vel::NTuple{2,Float64}              # 2D velocity of agent
    mass::Float64                       # Mass of agent to use in collisions
    type::Symbol                        # Cell type of agent
    color::String                       # Color used for agent in videos
    size::Int                           # Size of agent in videos
    num_interactions::Int               # Total number of interactions agent has
    antigens::Array                     # List of antigens the Tec agent contains
    age::Int                            # Age of agent; incremented after n steps
    just_aged::Bool                     # Boolean to determine if agent just aged (true) on current model step
end

mutable struct Thymocyte <: AbstractAgent
    id::Int                             # Unique ID to identify agent
    pos::NTuple{2,Float64}              # 2D position of agent
    vel::NTuple{2,Float64}              # 2D velocity of agent
    mass::Float64                       # Mass of agent to use in collisions
    type::Symbol                        # Cell type of agent
    color::String                       # Color used for agent in videos
    size::Int                           # Size of agent in videos
    num_interactions::Int               # Total number of interactions agent has
    tcr::String                         # TCR gene that the thymocyte agent is carrying
    age::Int                            # Age of agent; incremented after n steps
    just_aged::Bool                     # Boolean to determine if agent just aged (true) on current model step
    reaction_levels::Dict               # Dict to hold thymocyte's seen antigens and reaction levels to them
    autoreactive::Bool                  # Boolean to show if thymocyte is autoreactive (true) or not (false)
end

function initialize(;
    width_height = (1, 1),
    speed = 0.002,
    n_tecs = 50,
    n_thymocytes = 1000,
    dt=1.0,
    threshold = 0.8,
    autoreactive_proportion = 0.5,
    rng_seed = 42)

    rng = MersenneTwister(rng_seed)
    Random.seed!(rng)

    genes = [randstring(rng, 'A':'T', 9) for i=1:45] # 'A':'T' represents the 20 amino acids

    space2d = ContinuousSpace(width_height, 0.02)

    #interactions = zeros(n_tecs, n_thymocytes) # matrix to count thymocyte/tec interactions. cannot do this exactly if new agents are added over time - probably use some sort of list
    successful_interactions = 0
    unsuccessful_interactions = 0

    properties = @dict( # DrWatson @dict
        width_height,
        speed,
        dt,
        n_tecs,
        n_thymocytes,
        threshold,
        genes,
        #interactions,
        autoreactive_proportion,
        successful_interactions,
        unsuccessful_interactions,
        escaped_thymocytes = 0,
        autoreactive_thymocytes = 0,
        nonautoreactive_thymocytes = 0,
        total_thymocytes = 0,
        tecs_present = n_tecs) # ideally some easier, built-in way to keep track of this
    
    model = ABM(Union{Tec, Thymocyte}, space2d; properties, rng,)

    # Add agents to the model
    id = 0
    for _ in 1:n_tecs
        id += 1
        pos = Tuple(rand(model.rng, 2))
        vel = (0.0, 0.0)
        mass = Inf
        color = "#1f78b4"
        size = 40
        num_interactions = 0
        age = 0
        just_aged = false
        antis = sample(model.rng, genes, 2, replace = false) # choose a sample from genes to act as tec's antigens (replace = false to avoid repetitions, or are repetitions wanted?)
        tec = Tec(id, pos, vel, mass, :tec, color, size, num_interactions, antis, age, just_aged)
        add_agent!(tec, model)
    end
    for _ in 1:n_thymocytes
        id += 1
        model.total_thymocytes += 1
        pos = Tuple(rand(model.rng, 2))
        vel = sincos(2π * rand(model.rng)) .* model.speed
        mass = 1.0
        color = "#fdbf6f"
        size = 10
        num_interactions = 0
        age = 0
        just_aged = false
        reaction_levels = Dict{String, Float64}()
        if rand(model.rng) < autoreactive_proportion # randomly determine thymocyte reactivity according to match with autoreactive_proportion
            tcr = rand(model.rng, genes)
            auto = true
            model.autoreactive_thymocytes += 1
        else
            tcr = ""
            auto = false
            model.nonautoreactive_thymocytes += 1
        end
        thymocyte = Thymocyte(id, pos, vel, mass, :thymocyte, color, size, num_interactions, tcr, age, just_aged, reaction_levels, auto)
        add_agent!(thymocyte, model)
        set_color!(thymocyte, model)
    end
    return model
end

## Agent steps
function cell_move!(agent::Union{Tec, Thymocyte}, model)
    if agent.type == :thymocyte
        move_agent!(agent, model, model.dt)
        set_color!(agent, model)
        # add randomness to thymocytes' movements so they don't continue in same direction forever - same could be done by adding thymocyte collisions
        #walk!(agent, (rand(model.rng, -1.0:1.0),rand(model.rng, -1.0:1.0)), model) # might be unnecessary; breaks for spaces larger than (1, 1)
    end
end

function set_color!(agent::Union{Tec, Thymocyte}, model) # used to test accessing/modifying spatial properties
    if agent.pos[1] <= model.width_height[1] / 2 && agent.pos[2] <= model.width_height[2] / 2
        agent.color = "#FF0000"
    elseif agent.pos[1] <= model.width_height[1] / 2 && agent.pos[2] <= model.width_height[2]
        agent.color = "#00FF00"
    elseif agent.pos[1] <= model.width_height[1] && agent.pos[2] <= model.width_height[2] / 2
        agent.color = "#0000FF"
    else
        agent.color = "#FFFF00"
    end
end

## Model steps
function interact!(a1::Union{Tec, Thymocyte}, a2::Union{Tec, Thymocyte}, model)
    if a1.type == :tec && a2.type == :thymocyte # if tec/thymocyte collide, they can interact. relabel them here for simplicity
        tec_agent = a1
        thymocyte_agent = a2
    elseif a1.type == :thymocyte && a2.type == :tec
        tec_agent = a2
        thymocyte_agent = a1
    else
        return
    end

    tec_agent.num_interactions += 1
    thymocyte_agent.num_interactions += 1

    #model.interactions[tec_agent.id, thymocyte_agent.id - model.n_tecs] += 1 # subtract n_tecs from the thymocyte id to get thymocyte's matrix index

    if thymocyte_agent.tcr == "" # check if non-autoreactive thymocyte - is this necessary?
        #model.unsuccessful_interactions += 1 # not necessarily an interaction if thymocyte is non-autoreactive
        return
    else # compare a chosen tec antigen sequence to thymocyte TCR sequence
        # choose random antigen from tec's antigens to compare thymocyte tcr to. any matching characters in same position will increase reaction level to antigen.
        # if a reaction level passes the model threshold, the thymocyte is killed
        total_matches = 0 
        antigen = rand(model.rng, tec_agent.antigens)

        for i in range(1, length(antigen), step=1)
            if antigen[i] == thymocyte_agent.tcr[i]
                total_matches += 1
            end
        end
        
        reaction = total_matches / length(antigen) # strength of reaction

        if get(thymocyte_agent.reaction_levels, antigen, 0) != 0 # if thymocyte has seen antigen before, add to its current reaction level
            thymocyte_agent.reaction_levels[antigen] += reaction
        else # otherwise, add antigen as a new entry to the reaction_levels dict
            thymocyte_agent.reaction_levels[antigen] = reaction
        end

        if thymocyte_agent.reaction_levels[antigen] > model.threshold # kill thymocyte if sequence matches are above model threshold - (> or >=?)
            kill_agent!(thymocyte_agent, model)
            model.successful_interactions += 1
            model.autoreactive_thymocytes -= 1
        else
            model.unsuccessful_interactions += 1
        end
    end
end

function model_step!(model) # happens after every agent has acted
    interaction_radius = 0.06
    for (a1, a2) in interacting_pairs(model, interaction_radius, :types) # check :all versus :nearest versus :types. 
        #:types allows for easy changing of tec size by changing radius, but then thymocytes do not interact at all. :all causes error if thymocyte is deleted while having > 1 interaction. :nearest only allows for 1 interaction 
        if a1.num_interactions != 0 && a1.num_interactions % 20 == 0 # increment age after every 20 of agent's interactions
            a1.age += 1
            a1.just_aged = true
        end
        if a2.num_interactions != 0 && a2.num_interactions % 20 == 0
            a2.age += 1
            a2.just_aged = true
        end
        interact!(a1, a2, model)
        elastic_collision!(a1, a2, :mass)
    end

    if rand(model.rng) <= 0.002 # random chance to generate new thymocyte
        #model.n_thymocytes += 1
        pos = Tuple(rand(model.rng, 2))
        vel = sincos(2π * rand(model.rng)) .* model.speed
        if rand(model.rng) < model.autoreactive_proportion # set proportion of autoreactive vs non-autoreactive (empty string) thymocytes
            tcr = rand(model.rng, model.genes)
            auto = true
        else
            tcr = ""
            auto = false
        end
        thymocyte = Thymocyte(nextid(model), pos, vel, 1.0, :thymocyte, "#fdbf6f", 10, 0, tcr, 0, false, Dict(), auto)
        add_agent!(thymocyte, model)
        model.total_thymocytes += 1
    end

    if model.tecs_present < model.n_tecs # generate new tec if one dies
        model.tecs_present += 1
        pos = Tuple(rand(model.rng, 2))
        vel = (0.0, 0.0)
        antis = sample(model.rng, model.genes, 2, replace = false) # choose a sample from genes to act as tec's antigens (replace = false to avoid repetitions, or are repetitions wanted?)
        tec = Tec(nextid(model), pos, vel, Inf, :tec, "#1f78b4", 40, 0, antis, 0, false)
        add_agent!(tec, model)
    end

    for agent in allagents(model) # kill agent if it reaches certain age and update model properties depending on agent type/properties
        if (agent.age >= 14 && agent.type == :tec) || (agent.age >= 4 && agent.type == :thymocyte)
            kill_agent!(agent, model)
            if agent.type == :thymocyte
                if agent.autoreactive == true
                    model.escaped_thymocytes += 1
                    model.autoreactive_thymocytes -= 1
                else
                    model.nonautoreactive_thymocytes -= 1
                end
            else
                model.tecs_present -= 1
            end
        end

        if agent.type == :tec && agent.age != 0 && agent.age % 2 == 0 && agent.just_aged == true # add new antigen to list of tec antigens as it ages
            push!(agent.antigens, rand(model.rng, model.genes))
            agent.just_aged = false
        end
    end
end

cell_colors(a) = a.color
cell_sizes(a) = a.size
cell_markers(a) = a.type == :thymocyte ? :circle : :diamond

#mean_interactions_per_step(a) = mean(a) / a.steps # type error 

tec(a) = a.type == :tec
thymocyte(a) = a.type == :thymocyte

adata = [(tec, count), (thymocyte, count)]
alabels = ["tec count", "thymocyte count"]

react_ratio(model) = model.autoreactive_thymocytes/model.nonautoreactive_thymocytes # proportion of autoreactive_thymocytes to nonautoreactive_thymocytes - should decrease over time
escape_ratio(model) = model.escaped_thymocytes/model.total_thymocytes # proportion of escaped thymocutes to total thymocytes that appeared in simulation - should approach a constant

mdata = [:successful_interactions, :unsuccessful_interactions, escape_ratio, react_ratio]
mlabels = ["successful_interact count", "unsuccessful interact count", "escaped thymocytes", "reactivity_ratio"]

model2 = initialize(; width_height = (1, 1), n_tecs = 10, n_thymocytes = 1000, speed = 0.004, threshold = 0.75, autoreactive_proportion = 0.5, dt = 1, rng_seed = 42)

parange = Dict(:threshold => 0:0.01:1)

figure, adf, mdf = abm_data_exploration(
    model2, cell_move!, model_step!, parange;
    as = cell_sizes, ac = cell_colors, am = cell_markers, adata = adata, alabels = alabels,
    mdata = mdata, mlabels = mlabels)

#= abm_video(
    "thymus_abm2.mp4",
    model2,
    cell_move!,
    model_step!;
    frames = 1000,
    ac = cell_colors,
    as = cell_sizes,
    spf = 1,
    framerate = 20,
) =#

#= data, mdf = run!(model2, cell_move!, model_step!, 10000; adata = adata)

x = data.step
thy_data = data.count_thymocyte
tec_data = data.count_tec
figure = Figure(resolution = (600, 400))
ax = figure[1, 1] = Axis(figure, xlabel = "steps", ylabel = "Count")
lthy = lines!(ax, x, thy_data, color = :blue)
ltec = lines!(ax, x, tec_data, color = :red)
figure[1, 2] = Legend(figure, [lthy, ltec], ["Thymocytes", "Tecs"], textsize = 12)
display(figure) =#