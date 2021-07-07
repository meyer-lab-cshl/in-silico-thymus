using Agents
using Random
using InteractiveDynamics
using GLMakie # Needed for abm_data_exploration, CairoMakie does not work correctly with it but does work for plots/videos
using DrWatson: @dict
using Statistics: mean
using StatsBase # used to sample without replacement

mutable struct Tec <: AbstractAgent
    id::Int
    pos::NTuple{2,Float64}
    vel::NTuple{2,Float64}
    mass::Float64
    type::Symbol
    color::String
    size::Int
    num_interactions::Int
    antigens::Array
    age::Int
end

mutable struct Thymocyte <: AbstractAgent
    id::Int
    pos::NTuple{2,Float64}
    vel::NTuple{2,Float64}
    mass::Float64
    type::Symbol
    color::String
    size::Int
    num_interactions::Int
    tcr::String
    age::Int
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

    properties = @dict( # DrWatson @dict
        speed,
        dt,
        n_tecs,
        n_thymocytes,
        threshold,
        genes,
        #interactions,
        autoreactive_proportion)
    
    model = ABM(Union{Tec, Thymocyte}, space2d; properties, rng,)

    # Add agents to the model
    id = 0
    for _ in 1:n_tecs
        id += 1
        pos = Tuple(rand(model.rng, 2))
        vel = (0.0, 0.0)
        antis = sample(model.rng, genes, 2, replace = false) # choose a sample from genes to act as tec's antigens (replace = false to avoid repetitions, or are repetitions wanted?)
        tec = Tec(id, pos, vel, Inf, :tec, "#1f78b4", 40, 0, antis, 0)
        add_agent!(tec, model)
    end
    for _ in 1:n_thymocytes
        id += 1
        pos = Tuple(rand(model.rng, 2))
        vel = sincos(2π * rand(model.rng)) .* model.speed
        if rand(model.rng) < autoreactive_proportion # set proportion of autoreactive vs non-autoreactive (empty string) thymocytes
            tcr = rand(model.rng, genes)
        else
            tcr = ""
        end
        thymocyte = Thymocyte(id, pos, vel, 1.0, :thymocyte, "#fdbf6f", 10, 0, tcr, 0)
        add_agent!(thymocyte, model)
    end
    return model
end

## Agent steps
function cell_move!(agent::Union{Tec, Thymocyte}, model)
    if agent.type == :thymocyte
        move_agent!(agent, model, model.dt)
        # add randomness to thymocytes' movements so they don't continue in same direction forever - same could be done by adding thymocyte collisions
        #walk!(agent, (rand(model.rng, -1.0:1.0),rand(model.rng, -1.0:1.0)), model) # might be unnecessary; breaks for spaces larger than (1, 1)
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
        return
    else # compare a chosen tec antigen sequence to thymocyte TCR sequence
        total_matches = 0 
        antigen = rand(model.rng, tec_agent.antigens) # choose random antigen from tec's antigens to compare thymocyte tcr to

        for i in range(1, length(antigen), step=1)
            if antigen[i] == thymocyte_agent.tcr[i]
                total_matches += 1
            end
        end

        if total_matches / length(antigen) >= model.threshold # kill thymocyte if sequence matches are above model threshold
            kill_agent!(thymocyte_agent, model)
        end
    end
end

function model_step!(model) # happens after every agent has acted
    for (a1, a2) in interacting_pairs(model, 0.06, :types) # check :all versus :nearest versus :types. 
        #:types allows for easy changing of tec size by changing radius, but then thymocytes do not interact at all. :all causes error if thymocyte is deleted while having > 1 interaction. :nearest only allows for 1 interaction 
        interact!(a1, a2, model)
        elastic_collision!(a1, a2, :mass)

        if a1.num_interactions % 20 == 0 # increment age after every 20 of agent's interactions
            a1.age += 1
        end
        if a2.num_interactions % 20 == 0
            a2.age += 1
        end
    end

    if rand(model.rng) <= 0.002 # random chance to generate new thymocyte
        #model.n_thymocytes += 1
        pos = Tuple(rand(model.rng, 2))
        vel = sincos(2π * rand(model.rng)) .* model.speed
        if rand(model.rng) < model.autoreactive_proportion # set proportion of autoreactive vs non-autoreactive (empty string) thymocytes
            tcr = rand(model.rng, model.genes)
        else
            tcr = ""
        end
        thymocyte = Thymocyte(nextid(model), pos, vel, 1.0, :thymocyte, "#fdbf6f", 10, 0, tcr, 0)
        add_agent!(thymocyte, model)
    end

    if rand(model.rng) <= 0.002 # random chance to generate new tec
        #model.n_tecs += 1
        pos = Tuple(rand(model.rng, 2))
        vel = (0.0, 0.0)
        antis = sample(model.rng, model.genes, 2, replace = false) # choose a sample from genes to act as tec's antigens (replace = false to avoid repetitions, or are repetitions wanted?)
        tec = Tec(nextid(model), pos, vel, Inf, :tec, "#1f78b4", 40, 0, antis, 0)
        add_agent!(tec, model)
    end

    for agent in allagents(model) # kill agent if it reaches certain age
        if agent.age >= 5
            kill_agent!(agent, model)
        end
    end
end

model = initialize(; width_height = (1, 1), n_tecs = 5, n_thymocytes = 500, speed = 0.001, threshold = 0.8)
cell_colors(a) = a.color
cell_sizes(a) = a.size
cell_markers(a) = a.type == :thymocyte ? :circle : :diamond

#= abm_video(
    "thymus_abm.mp4",
    model,
    cell_move!,
    model_step!;
    frames = 1000,
    ac = cell_colors,
    as = cell_sizes,
    spf = 1,
    framerate = 20,
)
print(nagents(model) - model.n_tecs)
print(" thymocytes remaining") =#

#mean_interactions_per_step(a) = mean(a) / a.steps # type error 

tec(a) = a.type == :tec
thymocyte(a) = a.type == :thymocyte

adata = [(:num_interactions, mean, tec), (tec, count), (thymocyte, count)]
alabels = ["mean of total tec interactions", "tec count", "thymocyte count"]

#mdata = [nagents]
#mlabels = ["agent count"]

model2 = initialize(; width_height = (2, 2), n_tecs = 50, n_thymocytes = 500, speed = 0.002, threshold = 0.75, autoreactive_proportion = 0.5, dt = 1, rng_seed = 42)

parange = Dict(:threshold => 0:0.01:1)

figure, adf, mdf = abm_data_exploration(
    model2, cell_move!, model_step!, parange;
    as = cell_sizes, ac = cell_colors, am = cell_markers, adata = adata, alabels = alabels,)