using Distributions
using Random
using LinearAlgebra
using Plots

include("./src/Leabra.jl")

# This is an example network designed simply to be a pattern associator.
# This is pretty much the example from the Leabra Matlab code by Sergio Verduzco-Flores.

# 3 layers and their dimensions
dim_lays = [(5 , 5), 
            (10, 10), 
            (5 , 5)]

# connections[i,j] = c
# Layer i receives connections from layer j, and they have a relative strength c.
# The network constructor will normalize this matrix so that if there
# are non-zero entries in a row, they add to 1.            
# 2 receives connections from 1 at relative strength of 1
# 2 receives connections from 3 at relative strength of 0.2
# 3 receives connections from 2 at relative strength of 1
connections = [0  0  0;
               1  0 .2;
               0  1  0]

net = Leabra.build_network(dim_lays, connections)

n_inputs = 5

# inputs = Leabra.create_random_inputs(n_inputs) # this just creates random patterns for any size input/output layers
inputs = Leabra.create_patterns(n_inputs) # assumes a 5x5 input and output layer

n_epochs = 20
errors = Leabra.train_network!(net, inputs, n_epochs);

mean_errs = mean(errors, dims=2);
println("Mean errors: $mean_errs")
plot(mean_errs)
ylims!((0.0,1.0))