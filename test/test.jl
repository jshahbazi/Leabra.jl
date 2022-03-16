using Test

include("../src/Leabra.jl")

@testset "Leabra.jl tests" begin
    @test true == true


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

    @testset "Test Layer Functions" begin
        mylayer = Leabra.layer((1,1))
        Leabra.activities(mylayer)
        Leabra.acts_avg(mylayer)
        Leabra.scaled_acts(mylayer)
        Leabra.averages(mylayer)
        Leabra.rel_avg_l(mylayer)
        Leabra.updt_avg_l(mylayer)
        Leabra.updt_long_avgs(mylayer)
        Leabra.reset(mylayer)
    end

end