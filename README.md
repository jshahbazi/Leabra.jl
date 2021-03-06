# Leabra.jl

This is a Julia conversion of the old version of the Leabra algorithm written in Matlab by Sergio Verduzco.  The original code be found here: <https://github.com/emer/cemer/tree/master/Matlab>

The initial commits of this are almost an exact conversion, save for implementation of the NXX1 function which is based on the Go code found here: <https://github.com/emer/leabra>

An example network that can give you an idea of how to use this can be found in *example_network.jl*

# Installation

This code requires Julia 1.8.  It was developed with the beta version, but really its only using *const* for some structs.

It will install the following libraries into the project environment

- Distributions.jl
- LinearAlgebra.jl
- Plots.jl
- Random.jl

Ways to run the example:

- Clone this repo into a directory via `git clone https://github.com/jshahbazi/Leabra.jl`
- Run `cd Leabra.jl`
- Run `julia --project=. -e "using Pkg; Pkg.instantiate()"`
- Run `julia --project=. example_network.jl`

Or, you can run it from the Julia shell via these steps:

- Run `julia --project=.`
- From the `julia>` prompt, run: `include("example_network.jl")`

Or, just use the *example_notebook.ipynb*.  It has the code from *example_network.jl* in an easy-to-use format.

# Citation

Leabra and all associated ideas can be found here:

O’Reilly, R. C., Munakata, Y., Frank, M. J., Hazy, T. E., and Contributors (2012). Computational Cognitive Neuroscience. Wiki Book, 4th Edition (2020). URL: <https://CompCogNeuro.org>
