using Distributions
using Random
using LinearAlgebra
using Plots

include("./src/Leabra.jl")


dim_lays = [(5 , 5), 
            (10, 10), 
            (5 , 5)]
connections = [0  0  0;
                1  0 .2;
                0  1  0]

net = Leabra.build_network(dim_lays, connections)

n_inputs = 5
inputs = Leabra.create_random_inputs(n_inputs)

n_epochs = 10
n_trials = n_inputs
errors = Leabra.train_network!(net, inputs, n_epochs, n_trials)


mean_errs = mean(errors, dims=2);
println(mean_errs)
plot(mean_errs)