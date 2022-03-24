using Distributions
using Random
using LinearAlgebra

include("./src/Leabra.jl")

function build_network(dim_lays, connections)
    n_lays = length(dim_lays)
    n_units = zeros(Int64,1,n_lays)
    for i in 1:n_lays
        n_units[i] = dim_lays[i][1] * dim_lays[i][2]
    end
    w0 = Array{Any}(undef, (n_lays,n_lays))
    for rcv in 1:n_lays
        for snd in 1:n_lays
            if connections[rcv,snd] > 0
                w0[rcv,snd] = 0.3 .+ 0.4*rand(Uniform(),n_units[rcv],n_units[snd])
                # w0[rcv,snd] = rand(Uniform(),n_units[rcv],n_units[snd])
            else
                w0[rcv,snd] = 0.0
            end
        end
    end
    net = Leabra.network(dim_lays, connections, w0)

    return net
end


function create_random_inputs(n_inputs)
    inputs = Array{Array{Float64}, 2}(undef, (n_inputs,2))
    for i in 1:n_inputs
        inputs[i,1] = rand(Binomial(1,0.5),(n_inputs,n_inputs))
        inputs[i,2] = inputs[i,1];
    end

    return inputs
end

function train_network!(net, inputs, n_epochs, n_trials)
    n_minus = 50
    n_plus = 25

    lrate_sched = collect(LinRange(0.8, 0.2, n_epochs))
    errors = zeros(n_epochs,n_trials)
    for epoch in 1:n_epochs
        order = randperm(n_trials)
        net.lrate = lrate_sched[epoch]
        for trial in 1:n_trials
            Leabra.reset(net)
            pat = order[trial]

            # Minus phase
            input_array::Vector{Array{Float64}} = [inputs[pat, 1], [], []]
            for minus in 1:n_minus
                Leabra.cycle(net, input_array, true)
            end
            outs = (Leabra.activities(net.layers[3]))

            # Plus phase
            input_array = [inputs[pat, 1], [], inputs[pat, 2]];
            for plus in 1:n_plus
                Leabra.cycle(net, input_array, true);
            end
            Leabra.updt_long_avgs(net)

            # Learning
            Leabra.XCAL_learn(net)
            
            # Errors
            errors[epoch, pat] = 1 - sum(outs .* transpose(inputs[pat, 2][:])) / ( norm(outs) * norm(transpose(inputs[pat, 2][:])) );                    
        end
    end

    return errors
end





dim_lays = [(5 , 5), 
            (10, 10), 
            (5 , 5)]
connections = [0  0  0;
                1  0 .2;
                0  1  0]

net = build_network(dim_lays, connections)

n_inputs = 5
inputs = create_random_inputs(n_inputs)

n_epochs = 10
n_trials = n_inputs
errors = train_network!(net, inputs, n_epochs, n_trials)


mean_errs = mean(errors, dims=2);
println(mean_errs)


