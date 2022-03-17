using Test
using Distributions
using Random
using LinearAlgebra

include("../src/Leabra.jl")

@testset "Leabra.jl tests" begin

    @testset "Test Noisy XX1 Function" begin
        diff_tolerance = 1.0e-7

        test_values = [-0.05, -0.04, -0.03, -0.02, -0.01, 0, .01, .02, .03, .04, .05, .1, .2, .3, .4, .5]
        expected_results = [1.7735989e-14, 7.155215e-12, 2.8866178e-09, 1.1645374e-06, 0.00046864923, 0.094767615, 0.47916666, 0.65277773, 0.742268, 0.7967479, 0.8333333, 0.90909094, 0.95238096, 0.96774197, 0.9756098, 0.98039216]
        actual_results = Array{Float64}(undef, length(test_values))
    
        for i in 1:length(test_values)
            actual_results[i] = Leabra.NoisyXX1(test_values[i])
            diff = abs(actual_results[i] - expected_results[i])
            @test diff < diff_tolerance
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
        dim_lays = [(5,5), (10, 10), (5, 5)]
        connections = [0  0  0;
                    1  0 .2;
                    0  1  0];
        n_lays = length(dim_lays);
        n_units = zeros(Int64,1,n_lays); ## number of units in each layer
        for i in 1:n_lays
            n_units[i] = dim_lays[i][1] * dim_lays[i][2];
        end
        w0 = Array{Any}(undef, (n_lays,n_lays))
        for rcv in 1:n_lays
            for snd in 1:n_lays
                if connections[rcv,snd] > 0
                    w0[rcv,snd] = 0.3 .+ 0.4*rand(Uniform(),n_units[rcv],n_units[snd]);
                else
                    w0[rcv,snd] = 0.0
                end
            end
        end
        net = Leabra.network(dim_lays, connections, w0);
        n_inputs = 5;  # number of input-output patterns to associate
        patterns = Array{Matrix{Float64}, 2}(undef, (n_inputs,2)) # patterns{i,1} is the i-th input pattern, and 
                                                                # patterns{i,2} is the i-th output pattern.
        patterns[1,1] = repeat([0 0 1 0 0],5,1);   # vertical line
        patterns[2,1] = [1 1 1 1 1;zeros(4,5)];   # horizontal line
        patterns[3,1] = [0 0 0 0 1;0 0 0 1 0;0 0 1 0 0;0 1 0 0 0; 1 0 0 0 0]; # diagonal 1
        patterns[4,1] = reverse(patterns[3,1], dims=2) #patterns[3,1][[5 4 3 2 1],:]   # diagonal 2
        patterns[5,1] =  1.0 .* (!=(0.0).(patterns[3,1]) .|| !=(0.0).(patterns[4,1]))   # two diagonals
        for i in 1:n_inputs # outputs are the same as inputs (an autoassociator)
            patterns[i,1] = 0.01 .+ 0.98 .* patterns[i,1];
            patterns[i,2] = patterns[i,1];
        end
        n_epochs = 10;  # number of epochs. All input patterns are presented in one.
        n_trials = n_inputs; # number of trials. One input pattern per trial.
        n_minus = 50;  # number of minus cycles per trial.
        n_plus = 25; # number of plus cycles per trial.
        lrate_sched = collect(LinRange(0.8, 0.2, n_epochs))
        errors = zeros(n_epochs,n_trials); # cosine error for each pattern
        for epoch in 1:n_epochs
            order = randperm(n_trials); # order of presentation of inputs this epoch
            net.lrate = lrate_sched[epoch]; # learning rate for this epoch
            for trial in 1:n_trials
                Leabra.reset(net);  # randomize the acts for all units
                pat = order[trial];  # input to be presented this trial

                #++++++ MINUS PHASE +++++++
                inputs::Vector{Array{Float64}} = [patterns[pat, 1], [], []];
                for minus in 1:n_minus # minus cycles: layer 1 is clamped
                    Leabra.cycle(net, inputs, true);
                end
                outs = (Leabra.activities(net.layers[3])) # saving the output for testing

                #+++++++ PLUS PHASE +++++++
                inputs = [patterns[pat, 1], [], patterns[pat, 2]];
                for plus in 1:n_plus # plus cycles: layers 1 and 3 are clamped
                    Leabra.cycle(net, inputs, true);
                end
                Leabra.updt_long_avgs(net) # update averages used for net input scaling                            

                #+++++++ LEARNING +++++++
                Leabra.XCAL_learn(net)  # updates the avg_l vars and applies XCAL learning  
                
                #+++++++ ERRORS +++++++
                # Only the cosine error is used here
                errors[epoch, pat] = 1 - sum(outs .* transpose(patterns[pat, 2][:])) / ( norm(outs) * norm(transpose(patterns[pat, 2][:])) );                    
            end
        end

        mean_errs = mean(errors, dims=2);
        @test mean_errs == [0.6725562627407224; 0.30578951424903517; 0.08605332164946908; 0.03303809277458743; 0.010216950513064926; 0.0026514306318828674; 0.0021875149829591488; 0.0017011033595259796; 0.0017028339185111597; 0.001748068293656191;;]
    end

end