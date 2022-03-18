using Test
using Distributions
using Random
using LinearAlgebra

include("../src/Leabra.jl")

@testset "Leabra.jl tests" begin

    @testset "Test Noisy XX1 Function" begin
        test_values = [-0.05, -0.04, -0.03, -0.02, -0.01, 0, .01, .02, .03, .04, .05, .1, .2, .3, .4, .5]
        expected_results = [1.7735989e-14, 7.155215e-12, 2.8866178e-09, 1.1645374e-06, 0.00046864923, 0.094767615, 0.47916666, 0.65277773, 0.742268, 0.7967479, 0.8333333, 0.90909094, 0.95238096, 0.96774197, 0.9756098, 0.98039216]
        actual_results = Array{Float64}(undef, length(test_values))
    
        for i in 1:length(test_values)
            @test Leabra.NoisyXX1(test_values[i]) ≈ expected_results[i]  rtol=1e-5
        end        
    end

    @testset "Test Unit Functions" begin
        myunit = Leabra.Unit()
        Leabra.cycle(myunit, 0.8, 0.6)
        @test myunit.act == 0.1393939393939394
        @test myunit.adapt == 2.5145028932907716e-5
        Leabra.updt_avg_l(myunit)
        @test myunit.avg_l == 0.09015151515151515
        Leabra.clamped_cycle(myunit, 0.8)
        @test myunit.act == 0.8
        @test myunit.avg_ss == 0.48484848484848486
        @test myunit.avg_s == 0.33484848484848484
        @test myunit.avg_m == 0.21212121212121213
        Leabra.reset(myunit)
    end

    @testset "Test Layer Functions" begin
        mylayer = Leabra.layer((1,1))
        @test Leabra.acts_avg(mylayer) == 0.2
        @test Leabra.scaled_acts(mylayer) == [0.06666666666666667]
        @test Leabra.averages(mylayer) == ([0.2], [0.2], [0.1])
        @test Leabra.rel_avg_l(mylayer) == [0.0]
        @test Leabra.updt_long_avgs(mylayer) == 0.3333333333333333
        Leabra.reset(mylayer)

        mylayer = Leabra.layer((5,2))
        @test Leabra.acts_avg(mylayer) == 0.19999999999999998
        @test Leabra.scaled_acts(mylayer)[10] == 0.06666666666666667
        @test Leabra.averages(mylayer) == ([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], 
                                           [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], 
                                           [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        @test Leabra.rel_avg_l(mylayer) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        @test Leabra.updt_long_avgs(mylayer) == 0.25
        Leabra.reset(mylayer)
    end

    @testset "Test Network Functions" begin
        dim_lays = [(5 , 5), 
                    (10, 10), 
                    (5 , 5)]
        connections = [0  0  0;
                       1  0 .2;
                       0  1  0]
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
                else
                    w0[rcv,snd] = 0.0
                end
            end
        end
        net = Leabra.network(dim_lays, connections, w0)
        n_inputs = 5
        patterns = Array{Matrix{Float64}, 2}(undef, (n_inputs,2))
        patterns[1,1] = repeat([0 0 1 0 0],5,1)
        patterns[2,1] = [1 1 1 1 1;zeros(4,5)]
        patterns[3,1] = [0 0 0 0 1;0 0 0 1 0;0 0 1 0 0;0 1 0 0 0; 1 0 0 0 0]
        patterns[4,1] = reverse(patterns[3,1], dims=2)
        patterns[5,1] =  1.0 .* (!=(0.0).(patterns[3,1]) .|| !=(0.0).(patterns[4,1]))
        for i in 1:n_inputs
            patterns[i,1] = 0.01 .+ 0.98 .* patterns[i,1]
            patterns[i,2] = patterns[i,1]
        end
        n_epochs = 10
        n_trials = n_inputs
        n_minus = 50  # number of minus cycles per trial
        n_plus = 25   # number of plus cycles per trial
        lrate_sched = collect(LinRange(0.8, 0.2, n_epochs))
        errors = zeros(n_epochs,n_trials)
        for epoch in 1:n_epochs
            order = randperm(n_trials)
            net.lrate = lrate_sched[epoch]
            for trial in 1:n_trials
                Leabra.reset(net)
                pat = order[trial]

                # Minus phase
                inputs::Vector{Array{Float64}} = [patterns[pat, 1], [], []]
                for minus in 1:n_minus
                    Leabra.cycle(net, inputs, true)
                end
                outs = (Leabra.activities(net.layers[3]))

                # Plus phase
                inputs = [patterns[pat, 1], [], patterns[pat, 2]];
                for plus in 1:n_plus # plus cycles: layers 1 and 3 are clamped
                    Leabra.cycle(net, inputs, true);
                end
                Leabra.updt_long_avgs(net) # update averages used for net input scaling                            

                # Learning
                Leabra.XCAL_learn(net)  # updates the avg_l vars and applies XCAL learning  
                
                # Errors
                # Only the cosine error is used here
                errors[epoch, pat] = 1 - sum(outs .* transpose(patterns[pat, 2][:])) / ( norm(outs) * norm(transpose(patterns[pat, 2][:])) );                    
            end
        end

        mean_errs = mean(errors, dims=2);
        @test all(0.0 .< mean_errs) && all(mean_errs .< 1.0)
    end

end