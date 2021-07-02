using Agents
using Random
using InteractiveDynamics
using GLMakie # Needed for abm_data_exploration, CairoMakie does not work correctly with it but does work for plots/videos
using DrWatson: @dict
using Statistics: mean

mutable struct Tec <: AbstractAgent
    id::Int
    pos::NTuple{2,Float64}
    vel::NTuple{2,Float64}
    mass::Float64
    type::Symbol
    color::String
    size::Int
    num_interactions::Int
    antigen::String
    age::Int
    steps::Int
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
    steps::Int
end

function initialize(;
    width_height = (1, 1),
    speed = 0.002,
    n_tecs = 50,
    n_thymocytes = 1000,
    dt=1.0,
    threshold = 0.8,
    rng_seed = 42)

    rng = MersenneTwister(rng_seed)

    antigens = [randstring(rng, 'A':'Z', 9) for i=1:18000]

    space2d = ContinuousSpace(width_height, 0.02)

    interactions = zeros(n_tecs, n_thymocytes) # matrix to count thymocyte/tec interactions

    properties = @dict( # DrWatson @dict
        speed,
        dt,
        n_tecs,
        n_thymocytes,
        threshold,
        antigens,
        interactions)
    
    model = ABM(Union{Tec, Thymocyte}, space2d; properties, rng,)

    # Add agents to the model
    id = 0
    for _ in 1:n_tecs
        id += 1
        pos = Tuple(rand(model.rng, 2))
        vel = (0.0, 0.0)
        tec = Tec(id, pos, vel, Inf, :tec, "#1f78b4", 40, 0, rand(model.rng, antigens), 0, 0)
        add_agent!(tec, model)
    end
    for _ in 1:n_thymocytes
        id += 1
        pos = Tuple(rand(model.rng, 2))
        vel = sincos(2π * rand(model.rng)) .* model.speed
        if rand(model.rng) > 0.5 # set proportion of autoreactive vs non-autoreactive (empty string) thymocytes - probably add as model parameter
            tcr = rand(model.rng, antigens)
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
        walk!(agent, (rand(model.rng, -1.0:1.0),rand(model.rng, -1.0:1.0)), model) 
    end
    agent.steps += 1
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

    model.interactions[tec_agent.id, thymocyte_agent.id - model.n_tecs] += 1 # subtract n_tecs from the thymocyte id to get thymocyte's matrix index

    if thymocyte_agent.tcr == "" # check if non-autoreactive thymocyte - is this necessary?
        return
    else
        total_matches = 0 # compare tec antigen sequence to thymocyte TCR sequence
#=         for i in tec_agent.antigen
            for j in thymocyte_agent.tcr
                if i == j
                    total_matches += 1
                end
            end
        end =#

        for i in range(1, length(tec_agent.antigen), step=1)
            if tec_agent.antigen[i] == thymocyte_agent.tcr[i]
                total_matches += 1
            end
        end

        if total_matches / length(tec_agent.antigen) >= model.threshold # kill thymocyte if sequence matches are above model threshold
            kill_agent!(thymocyte_agent, model)
        end
    end
end

function model_step!(model) # happens after every agent has acted
    for (a1, a2) in interacting_pairs(model, 0.06, :types) # check :all versus :nearest versus :types. 
        #:types allows for easy changing of tec size by changing radius, but then thymocytes do not interact at all. :all causes error if thymocyte is deleted while having > 1 interaction. :nearest only allows for 1 interaction 
        interact!(a1, a2, model)
        elastic_collision!(a1, a2, :mass)
    end
    #if rand(model.rng) <= 0.02 # random chance to generate new thymocyte
    #    model.n_thymocytes += 1
    #    pos = Tuple(rand(model.rng, 2))
    #    vel = sincos(2π * rand(model.rng)) .* model.speed
    #    thymocyte = Thymocyte(nextid(model), pos, vel, 1.0, :thymocyte, "#fdbf6f", 0, 0)
    #    add_agent!(thymocyte, model)
    #end
end

model = initialize(; width_height = (1, 1), n_tecs = 5, n_thymocytes = 500, speed = 0.001, threshold = 0.8)
cell_colors(a) = a.color
cell_sizes(a) = a.size

abm_video(
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
print(" thymocytes remaining")

is_tec(a) = a.type == :tec
#mean_interactions_per_step(a) = mean(a) / a.steps # type error 

tec(a) = a.type == :tec
thymocyte(a) = a.type == :thymocyte

adata = [(:num_interactions, mean, is_tec), (tec, count), (thymocyte, count)]
alabels = ["mean of total tec interactions", "tec count", "thymocyte count"]

#mdata = [nagents]
#mlabels = ["agent count"]

model2 = initialize(; width_height = (1, 1), n_tecs = 5, n_thymocytes = 500, speed = 0.002, threshold = 0.75)

parange = Dict(:threshold => 0:0.01:1)

figure, adf, mdf = abm_data_exploration(
    model2, cell_move!, model_step!, parange;
    as = cell_sizes, ac = cell_colors, adata = adata, alabels = alabels,)