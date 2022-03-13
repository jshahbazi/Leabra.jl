module Leabra

using BenchmarkTools
using ProfileVega
using Distributions
using DataStructures
using JLD2
using LinearAlgebra
using Parameters
using Plots
using Random
using Statistics

import Base.Threads.@spawn

######################################################################################################
# Utility Functions
#
"""
Prints variables and arrays in a more structured form that's easier to read
"""
function pretty_print(variable, prefix="")
    print(prefix)
    show(IOContext(stdout, :limit => false), "text/plain", variable)
    println()
end

"""
Expands an array along a specific dimension for a specific number of times
Found here: https://stackoverflow.com/questions/62520884/expanding-array-along-a-dimension
"""
function expand(x, dim, copies)
    sz = size(x)
    rep = ntuple(d->d==dim ? copies : 1, length(sz)+1)
    new_size = ntuple(d->d<dim ? sz[d] : d == dim ? 1 : sz[d-1], length(sz)+1)
    return repeat(reshape(x, new_size), outer=rep)
end

######################################################################################################








end # module Leabra
