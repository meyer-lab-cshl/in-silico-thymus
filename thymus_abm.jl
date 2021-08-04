using Base: Float64
using Agents: length, isempty, getindex
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
    death_label::Bool                   # Label a thymocyte to be killed on its next step
    confined::Bool
    bind_location::NTuple{2,Float64}
    treg::Bool
end

function initialize(;
    width_height = (1, 1),
    speed = 0.002,
    n_tecs = 50,
    n_thymocytes = 1000,
    dt=1.0,
    threshold = 0.8,
    treg_threshold = 0.6,
    autoreactive_proportion = 0.5,
    rng_seed = 42)

    rng = MersenneTwister(rng_seed)

    genes = [randstring(rng, 'A':'T', 9) for i=1:45] # 'A':'T' represents the 20 amino acids

    space2d = ContinuousSpace(width_height, 0.02)

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
        autoreactive_proportion,
        successful_interactions,
        unsuccessful_interactions,
        escaped_thymocytes = 0,
        autoreactive_thymocytes = 0,
        nonautoreactive_thymocytes = 0,
        total_thymocytes = 0,
        treg_threshold,
        num_tregs = 0,
        tecs_present = n_tecs) # ideally some easier, built-in way to keep track of this
    
    model = ABM(Union{Tec, Thymocyte}, space2d; properties, rng,)

    # Add agents to the model
    id = 0
    for _ in 1:n_tecs
        id += 1
        pos = Tuple(rand(model.rng, 2))
        if !isempty(nearby_ids(pos, model, model.width_height[1]/2)) # make sure mTECs don't overlap
            pos = Tuple(rand(model.rng, 2))
        end
        vel = (0.0, 0.0)
        mass = Inf
        color = "#00ffff"
        size = 90
        num_interactions = 0
        age = rand(model.rng, 0:19) # randomize starting ages
        just_aged = false
        antis = sample(model.rng, genes, age+2, replace = false) # choose an (age+2) size sample from genes to act as tec's antigens (replace = false to avoid repetitions, or are repetitions wanted?)
        tec = Tec(id, pos, vel, mass, :tec, color, size, num_interactions, antis, age, just_aged)
        add_agent!(tec, model)
        set_color!(tec)
    end

    autoreactive_counter = 0
    for _ in 1:n_thymocytes
        id += 1
        model.total_thymocytes += 1
        pos = Tuple(rand(model.rng, 2))
        vel = sincos(2π * rand(model.rng)) .* model.speed
        mass = 1.0
        color = "#006400"
        size = 15
        num_interactions = 0
        age = rand(model.rng, 0:3) # randomize starting ages
        just_aged = false
        reaction_levels = Dict{String, Float64}()
        if autoreactive_counter < n_thymocytes * autoreactive_proportion # randomly determine thymocyte reactivity according to match with autoreactive_proportion
            tcr = rand(model.rng, genes)
            auto = true
            model.autoreactive_thymocytes += 1
            autoreactive_counter += 1
        else
            tcr = ""
            auto = false
            model.nonautoreactive_thymocytes += 1
        end
        death_label = false
        confined = false
        bind_location = (0.0,0.0)
        treg = false
        thymocyte = Thymocyte(id, pos, vel, mass, :thymocyte, color, size, num_interactions, tcr, age, just_aged, reaction_levels, auto, death_label, confined, bind_location, treg)
        add_agent!(thymocyte, model)
        set_color!(thymocyte)
    end
    return model
end

## Agent steps
function cell_move!(agent::Union{Tec, Thymocyte}, model)
    if agent.type == :thymocyte
        if agent.death_label == true # maybe weird to take care of agent death here, but doing it in interact! in model_step! sometimes causes key value errors - does this introduce any problems?
            kill_agent!(agent, model)
            return
            # Fix movement under confinement below? - some agents move back and forth over short distance
        elseif agent.confined == true
            if agent.pos[1] >= agent.bind_location[1] + 0.3 || agent.pos[2] >= agent.bind_location[2] + 0.3 || agent.pos[1] >= agent.bind_location[1] - 0.3 || agent.pos[2] >= agent.bind_location[2] - 0.3
                if get_direction(agent.pos, agent.bind_location, model)[1] < 0 || get_direction(agent.pos, agent.bind_location, model)[2] < 0 
                    agent.vel = -1 .* agent.vel
                end
            end
            move_agent!(agent, model, model.dt)
        else
            move_agent!(agent, model, model.dt)
        end
        # add randomness to thymocytes' movements so they don't continue in same direction forever - maybe too much?
        walk!(agent, (rand(model.rng, -model.width_height[1]:model.width_height[2]) .* model.speed,rand(model.rng, -model.width_height[1]:model.width_height[2]) .* model.speed), model)
    end
    #set_color!(agent)
end

function set_color!(agent::Union{Tec, Thymocyte})
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
    else
        if agent.confined == true
            agent.color = "#ffff00"
        elseif agent.autoreactive == true
            agent.color = "#ff0000"
#=         else
        #if agent.age == 1
        #    agent.color = "#ffff00"
        #elseif agent.age == 2
        #    agent.color = "#ffa500"
            if agent.age == 3
                agent.color = "#ff0000"
            elseif agent.age == 4
                agent.color = "#7f00ff"
            end =#
        end
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

    thymocyte_agent.num_interactions += 1

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

        if thymocyte_agent.reaction_levels[antigen] >= model.threshold # kill thymocyte if sequence matches are above model threshold - (> or >=?)
            #kill_agent!(thymocyte_agent, model)
            if rand(model.rng) > 0.5
                thymocyte_agent.death_label = true
                model.autoreactive_thymocytes -= 1
            else
                thymocyte_agent.confined = true
                thymocyte_agent.bind_location = thymocyte_agent.pos
                thymocyte_agent.vel = 0.5 .* thymocyte_agent.vel
            end
            model.successful_interactions += 1
            tec_agent.num_interactions += 1 # only increment if successful interaction w/ a reactive thymocyte?
        else
            if thymocyte_agent.reaction_levels[antigen] >= model.treg_threshold && thymocyte_agent.reaction_levels[antigen] < model.threshold
                thymocyte_agent.treg = true
            end
            model.unsuccessful_interactions += 1
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

function model_step!(model) # happens after every agent has acted
    interaction_radius = 0.05*model.width_height[1]
    for (a1, a2) in interacting_pairs(model, interaction_radius, :types) # check :all versus :nearest versus :types.
        #:types allows for easy changing of tec size by changing radius, but then thymocytes do not interact at all. :all causes error if thymocyte is deleted while having > 1 interaction. :nearest only allows for 1 interaction 
        interact!(a1, a2, model)
        elastic_collision!(a1, a2, :mass)
    end

    if rand(model.rng) <= 0.15 # random chance to generate new thymocyte
        pos = Tuple(rand(model.rng, 2))
        vel = sincos(2π * rand(model.rng)) .* model.speed
        if rand(model.rng) < model.autoreactive_proportion # set proportion of autoreactive vs non-autoreactive (empty string) thymocytes
            tcr = rand(model.rng, model.genes)
            auto = true
            model.autoreactive_thymocytes += 1
        else
            tcr = ""
            auto = false
            model.nonautoreactive_thymocytes += 1
        end
        thymocyte = Thymocyte(nextid(model), pos, vel, 1.0, :thymocyte, "#006400", 15, 0, tcr, 0, false, Dict(), auto, false, false, (0.0,0.0), false)
        add_agent!(thymocyte, model)
        model.total_thymocytes += 1
    end

    if model.tecs_present < model.n_tecs # generate new tec if one dies
        model.tecs_present += 1
        overlap = true
        while overlap == true # should be good to make sure new tecs don't overlap an existing tec - could be infinite loop or very long as less positions are available - though should never infinite loop if always replacing a dead one
            pos = Tuple(rand(model.rng, 2))
            for a in nearby_ids(pos, model, model.width_height[1]/2)
                if getindex(model, a).type == :tec
                    break
                end
                overlap = false
            end
        end
        vel = (0.0, 0.0)
        antis = sample(model.rng, model.genes, 2, replace = false) # choose a sample from genes to act as tec's antigens (replace = false to avoid repetitions, or are repetitions wanted?)
        tec = Tec(nextid(model), pos, vel, Inf, :tec, "#00c8ff", 90, 0, antis, 0, false)
        add_agent!(tec, model)
    end

    for agent in allagents(model) # kill agent if it reaches certain age and update model properties depending on agent type/properties
        if (agent.age >= 20 && agent.type == :tec) || (agent.age >= 4 && agent.type == :thymocyte)
            if agent.type == :thymocyte
                if agent.autoreactive == true
                    model.escaped_thymocytes += 1
                    model.autoreactive_thymocytes -= 1
                else
                    model.nonautoreactive_thymocytes -= 1
                end

                if agent.treg == true
                    model.num_tregs += 1
                end
            else
                model.tecs_present -= 1
            end
            kill_agent!(agent, model)
        end

        if agent.type == :tec && agent.age != 0 && agent.just_aged == true # add new antigen to list of tec antigens as it ages
            push!(agent.antigens, rand(model.rng, model.genes))
            agent.just_aged = false
        end
    end
end

cell_colors(a) = a.color
cell_sizes(a) = a.size
cell_markers(a) = a.type == :thymocyte ? :circle : :diamond

tec(a) = a.type == :tec
thymocyte(a) = a.type == :thymocyte

#adata = [(tec, count), (thymocyte, count)]
#alabels = ["tec count", "thymocyte count"]
adata = [(thymocyte, count)]
alabels = ["thymocyte count"]

react_ratio(model) = model.autoreactive_thymocytes/model.nonautoreactive_thymocytes # proportion of autoreactive_thymocytes to nonautoreactive_thymocytes - should decrease over time
escape_ratio(model) = model.escaped_thymocytes/model.total_thymocytes # proportion of escaped thymocutes to total thymocytes that appeared in simulation - should approach a constant

mdata = [:num_tregs, :successful_interactions, :unsuccessful_interactions, escape_ratio, react_ratio]
mlabels = ["number of tregs", "successful interactions ", "unsuccessful interactions", "escaped thymocytes", "reactivity_ratio"]

model2 = initialize(; width_height = (2, 2), n_tecs = 10, n_thymocytes = 1000, speed = 0.005, threshold = 0.75, autoreactive_proportion = 0.5, dt = 1, rng_seed = 42, treg_threshold = 0.6)

parange = Dict(:threshold => 0:0.01:1)

figure, adf, mdf = abm_data_exploration(
    model2, cell_move!, model_step!, parange;
    as = cell_sizes, ac = cell_colors, am = cell_markers, adata = adata, alabels = alabels,
    mdata = mdata, mlabels = mlabels)

#= abm_video(
    "thymus_abm_vid.mp4",
    model2,
    cell_move!,
    model_step!;
    frames = 500,
    ac = cell_colors,
    as = cell_sizes,
    spf = 1,
    framerate = 20,
)
 =#
#= data, mdf = run!(model2, cell_move!, model_step!, 1000; adata = adata)

x = data.step
thy_data = data.count_thymocyte
tec_data = data.count_tec
figure = Figure(resolution = (600, 400))
ax = figure[1, 1] = Axis(figure, xlabel = "steps", ylabel = "Count")
lthy = lines!(ax, x, thy_data, color = :blue)
ltec = lines!(ax, x, tec_data, color = :red)
figure[1, 2] = Legend(figure, [lthy, ltec], ["Thymocytes", "Tecs"], textsize = 12)
display(figure) =#