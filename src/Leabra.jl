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
# Unit Functions
#
mutable struct Unit 
    act::Float64        # = 0.2           # "firing rate" of the unit
    avg_ss::Float64     # = act           # super short-term average of act
    avg_s::Float64      # = act           # short-term average of act
    avg_m::Float64      # = act           # medium-term average of act
    avg_l::Float64      # = 0.1           # long-term average of act    
    net::Float64        # = 0.0           # net input. Asymptotically approaches net_raw (see cycle).
    v_m::Float64        # = 0.3           # membrane potential
    vm_eq::Float64      # = 0.3           # a version of v_m that doesn't reset with spikes
    adapt::Float64      # = 0.0           # adaptation, as in the AdEx model 
    spike::Bool         # = false         # a flag that indicates spiking threshold was crossed

    # constants
    const net_dt::Float64     # = 1/1.4         # time step constant for update of 'net'
    const integ_dt::Float64   # = 1.0;    # time step constant for integration of cycle dynamics
    const vm_dt::Float64      # = 1/3.3;  # time step constant for membrane potential
    const l_dn_dt::Float64    # = 1/2.5;  # time step constant for avg_l decrease
    const adapt_dt::Float64   # = 1/144;  # time step constant for adaptation
    const ss_dt::Float64      # = 0.5;    # time step for super-short average
    const s_dt::Float64       # = 0.5;    # time step for short average
    const m_dt::Float64       # = 0.1;    # time step for medium-term average
    const avg_l_dt::Float64   # = 0.1;    # time step for long-term average
    const avg_l_max::Float64  # = 1.5;          # max value of avg_l
    const avg_l_min::Float64  # = 0.1;          # min value of avg_l
    const e_rev_e::Float64    # = 1.0;          # excitatory reversal potential
    const e_rev_i::Float64    # = 0.25;         # inhibitory reversal potential
    const e_rev_l::Float64    # = 0.3;          # leak reversal potential
    const gc_l::Float64       # = 0.1;          # leak conductance
    const thr::Float64        # = 0.5;          # normalized "rate threshold"
    const spk_thr::Float64    # = 1.2;          # normalized spike threshold
    const vm_r::Float64       # = 0.3;          # reset membrane potential after spike
    const vm_gain::Float64    # = 0.04;         # gain that voltage produces on adaptation
    const spike_gain::Float64 # = 0.00805;      # effect of spikes on adaptation
    const l_up_inc::Float64   # = 0.2;          # increase in avg_l if avg_m has been 'large'   
    
    function Unit()
        return new( 0.2, 0.2, 0.2, 0.2, 0.1, 0.0, 0.3, 0.3, 0.0, false,
                    1/1.4, 1.0, 1/3.3, 1/2.5, 1/144, 0.5, 0.5, 0.1, 0.1, 1.5, 0.1, 1.0, 0.25, 0.3, 0.1, 0.5, 1.2, 0.3, 0.04, 0.00805, 0.2)
     end    
end
    
function rel_avg_l(u::Unit)::Float64
    return (u.avg_l - u.avg_l_min)/(u.avg_l_max - u.avg_l_min)
end


function cycle(u::Unit, net_raw::Float64, gc_i::Float64)
    # Does one Leabra cycle. Called by the layer cycle method.
    # net_raw = instantaneous, scaled, received input
    # gc_i = fffb inhibition
    
    ## updating net input
    u.net = u.net + u.integ_dt * u.net_dt * (net_raw - u.net)
    
    ## Finding membrane potential
    I_net = u.net*(u.e_rev_e - u.v_m) + u.gc_l*(u.e_rev_l - u.v_m) + gc_i*(u.e_rev_i - u.v_m)
    # almost half-step method for updating v_m (adapt doesn't half step)
    v_m_h = u.v_m + 0.5*u.integ_dt*u.vm_dt*(I_net - u.adapt)
    I_net_h = u.net*(u.e_rev_e - v_m_h) + u.gc_l*(u.e_rev_l - v_m_h) + gc_i*(u.e_rev_i - v_m_h)
    u.v_m = u.v_m + u.integ_dt*u.vm_dt*(I_net_h - u.adapt)
    u.vm_eq = u.vm_eq + u.integ_dt*u.vm_dt*(I_net_h - u.adapt)
    
    ## Finding activation
    # finding threshold excitatory conductance
    g_e_thr = (gc_i*(u.e_rev_i-u.thr) + u.gc_l*(u.e_rev_l-u.thr) - u.adapt) / (u.thr - u.e_rev_e)
    # finding whether there's an action potential
    if u.v_m > u.spk_thr
        u.spike = true
        u.v_m = u.vm_r
    else
        u.spike = false
    end
    # finding instantaneous rate due to input

    if u.vm_eq <= u.thr
        new_act = nxx1(u.vm_eq - u.thr)[1]
    else
        new_act = nxx1(u.net - g_e_thr)[1]
    end

    # update activity
    u.act = u.act + u.integ_dt * u.vm_dt * (new_act - u.act)

    ## Updating adaptation
    u.adapt = u.adapt + u.integ_dt*(u.adapt_dt*(u.vm_gain*(u.v_m - u.e_rev_l)  - u.adapt) + u.spike*u.spike_gain)
          
    ## updating averages
    u.avg_ss = u.avg_ss + u.integ_dt * u.ss_dt * (u.act - u.avg_ss)
    u.avg_s = u.avg_s + u.integ_dt * u.s_dt * (u.avg_ss - u.avg_s)
    u.avg_m = u.avg_m + u.integ_dt * u.m_dt * (u.avg_s - u.avg_m) 
end

function clamped_cycle(u::Unit, input)
    # This function performs one cycle of the unit when its activty
    # is clamped to an input value. The activity is set to be equal
    # to the input, and all the averages are updated accordingly.
    
    ## Clamping the activty to the input
    u.act = input;
    
    ## updating averages
    u.avg_ss = u.avg_ss + u.integ_dt * u.ss_dt * (u.act - u.avg_ss);
    u.avg_s = u.avg_s + u.integ_dt * u.s_dt * (u.avg_ss - u.avg_s);
    u.avg_m = u.avg_m + u.integ_dt * u.m_dt * (u.avg_s - u.avg_m);
end


function updt_avg_l(u::Unit)
    # This fuction updates the long-term average 'avg_l' 
    # u = this unit
    # Based on the description in:
    # https://grey.colorado.edu/ccnlab/index.php/Leabra_Hog_Prob_Fix#Adaptive_Contrast_Impl
    
    if u.avg_m > 0.2
        u.avg_l = u.avg_l + u.avg_l_dt*(u.avg_l_max - u.avg_m)
    else
        u.avg_l = u.avg_l + u.avg_l_dt*(u.avg_l_min - u.avg_m)
    end
end

function reset(u::Unit)
    # This function sets the activity to a random value, and sets
    # all activity time averages equal to that value.
    # Used to begin trials from a random stationary point.
    u.act = 0.0
    u.avg_ss = u.act
    u.avg_s = u.act
    u.avg_m = u.act
    u.avg_l = u.act
    u.net = 0.0
    u.v_m = 0.3
    u.vm_eq = 0.3
    u.adapt = 0.0            
    u.spike = 0.0            
end

######################################################################################################
# NXX1 Functions
#

function XX1(x::Float64)
    return x / (x + 1)
end

function XX1GainCor(x::Float64, GainCorRange, NVar, Gain, GainCor)
    gainCorFact = (GainCorRange - (x / NVar)) / GainCorRange
    if gainCorFact < 0
        return XX1(Gain * x)
    end
    newGain = Gain * (1 - GainCor*gainCorFact)
    return XX1(newGain * x)
end   

struct NXX1Parameters
    Thr::Float64            # = 0.5
    Gain::Int64             # = 100
    NVar::Float64           # = 0.005
    VmActThr::Float64       # = 0.01
    SigMult::Float64        # = 0.33
    SigMultPow::Float64     # = 0.8
    SigGain::Float64        # = 3.0
    InterpRange::Float64    # = 0.01
    GainCorRange::Float64   # = 10.0
    GainCor::Float64        # = 0.1

	SigGainNVar::Float64    # = SigGain / NVar
	SigMultEff::Float64     # = SigMult * ((Gain* NVar) ^ SigMultPow)
	SigValAt0::Float64      # = 0.5 * SigMultEff
	InterpVal::Float64      # = XX1GainCor(InterpRange, GainCorRange, NVar, Gain, GainCor) - SigValAt0

    function NXX1Parameters()
        return new(0.5, 100, 0.005, 0.01, 0.33, 0.8, 3.0, 0.01, 10.0, 0.1, 
                   3.0/0.005, (0.33 * ((100*0.005)^0.8)), 0.5 * (0.33 * ((100*0.005)^0.8)), 
                   XX1GainCor(0.01,10.0,0.005,100,0.1) - (0.5 * (0.33 * ((100*0.005)^0.8))) )
    end
end

nxx1p = NXX1Parameters() # default set of parameters to pass in to functions

function XX1GainCor(x::Float64, xp::NXX1Parameters = nxx1p)
    gainCorFact = (xp.GainCorRange - (x / xp.NVar)) / xp.GainCorRange
    if gainCorFact < 0
        return XX1(xp.Gain * x)
    end
    newGain = xp.Gain * (1 - xp.GainCor * gainCorFact)
    return XX1(newGain * x)
end  

function NoisyXX1(x::Float64, xp::NXX1Parameters = nxx1p)
	if x < 0
		return xp.SigMultEff / (1 + exp(-(x * xp.SigGainNVar)))
	elseif x < xp.InterpRange
		interp = 1 - ((xp.InterpRange - x) / xp.InterpRange)
		return xp.SigValAt0 + interp * xp.InterpVal
	else
		return XX1GainCor(x)
    end
end

function XX1GainCorGain(x::Float64, gain::Float64, xp::NXX1Parameters = nxx1p)
	gainCorFact = (xp.GainCorRange - (x / xp.NVar)) / xp.GainCorRange
	if gainCorFact < 0
		return XX1(gain * x)
    end
	newGain = gain * (1 - xp.GainCor * gainCorFact)
	return XX1(newGain * x)
end

function NoisyXX1Gain(x::Float64, gain::Float64, xp::NXX1Parameters = nxx1p)
	if x < xp.InterpRange
		sigMultEffArg = xp.SigMult * (gain * xp.NVar ^ xp.SigMultPow)
		sigValAt0Arg = 0.5 * sigMultEffArg

		if x < 0 # sigmoidal for < 0
			return sigMultEffArg / (1 + exp(-(x * xp.SigGainNVar)))
		else # else x < interp_range
			interp = 1 - ((xp.InterpRange - x) / xp.InterpRange)
			return sigValAt0Arg + interp * xp.InterpVal
        end
	else
		return xp.XX1GainCorGain(x, gain)
    end
end

function nxx1(points, xp::NXX1Parameters = nxx1p)
    results = Array{Float64}(undef, length(points))
    for (index,value) in enumerate(points)
        results[index] = NoisyXX1(value)
    end
    return results
end

######################################################################################################
#  Layer Functions
#

mutable struct Layer 
    units::Array{Unit}
    pct_act_scale::Float64
    acts_p_avg::Float64
    netin_avg::Float64
    wt::Array{Float64, 2}
    ce_wt::Array{Float64, 2}
    N::Int64
    fbi::Float64

    const ff::Float64           # = 1.0
    const ff0::Float64          # = 0.1
    const fb::Float64           # = 0.5
    const fb_dt::Float64        # = 1/1.4 # time step for fb inhibition (fb_tau=1.4)
    const gi::Float64           # = 2.0
    const avg_act_dt::Float64   # = 0.01
end

function layer(dims::Tuple{Int64, Int64} = (1,1))
    N = dims[1]*dims[2]
    # lay.units = unit.empty;  # so I can make the assignment below
    # lay.units(lay.N,1) = unit; # creating all units
    # # notice how the unit array is 1-D. The 2-D structure of the
    # # layer doesn't have meaning in this part of the code 
    
    units = [Unit() for i in 1:N]
    
    acts_avg = 0.2 # should be the average but who knows
    avg_act_n =  1.0 # should be the avg
    pct_act_scale = 1/(avg_act_n + 2);
    fb = 0.5

    lay = Layer(units, 
                pct_act_scale, 
                acts_avg, 
                0,
                zeros(Float64, (1,1)),
                zeros(Float64, (1,1)),
                N,
                fb * acts_avg,
                1.0, 0.1, fb, 1/1.4, 2.0, 0.01
    )

    return lay
end

function acts_avg(lay::Layer)
    # get the value of acts_avg, the mean of unit activities
    return mean(activities(lay));
end

function activities(lay::Layer)
    # ## returns a vector with the activities of all units
    # acts = Array{Float64, 1}(undef, lay.N)
    acts = zeros(Float64, lay.N)
    for (index,unit) in enumerate(lay.units)
        acts[index] = unit.act
    end
    return transpose(acts)
end

function scaled_acts(lay::Layer)
    # ## returns a vector with the scaled activities of all units
    acts = Array{Float64, 1}(undef, lay.N)
    for (index,unit) in enumerate(lay.units)
        acts[index] = unit.act
    end
    return lay.pct_act_scale .* acts #collect(transpose(acts))
end

function cycle(lay::Layer, raw_inputs::Array{Float64}, ext_inputs::Array{Float64})
    ## this function performs one Leabra cycle for the layer
    #raw_inputs = An Ix1 matrix, where I is the total number of inputs
    #             from all layers. Each input has already been scaled
    #             by the pct_act_scale of its layer of origin and by
    #             the wt_scale_rel factor.
    #ext_inputs = An Nx1 matrix denoting inputs that don't come
    #             from another layer, where N is the number of
    #             units in this layer. An empty matrix indicates
    #             that there are no external inputs.
     
    ## obtaining the net inputs            
    netins = lay.ce_wt * raw_inputs;  # you use contrast-enhanced weights
    if any(ext_inputs) .> 0.0
        netins = netins .+ ext_inputs;
    end

    ## obtaining inhibition
    lay.netin_avg = mean(netins); 
    ffi = lay.ff * max(lay.netin_avg - lay.ff0, 0);
    lay.fbi = lay.fbi + lay.fb_dt * (lay.fb * acts_avg(lay) - lay.fbi);
    gc_i = lay.gi * (ffi + lay.fbi); 
    
    ## calling the cycle method for all units
    # function cycle(u::Unit, net_raw::Float64, gc_i::Float64)
    for i in 1:lay.N  # a parfor here?
        cycle(lay.units[i], netins[i], gc_i);
    end
end

function clamped_cycle(lay::Layer, input::Array{Float64})
    # sets all unit activities equal to the input and updates all
    # the variables as in the cycle function.
    # input = vector specifying the activities of all units
    for i = 1:lay.N  # parfor ?
        # function clamped_cycle(u::Unit, input)
        clamped_cycle(lay.units[i], input[i]);
    end
    lay.fbi = lay.fb * acts_avg(lay)
end
        
function averages(lay::Layer)
    # Returns the s,m,l averages in the layer as vectors.
    # Notice that the ss average is not returned, and avg_l is not
    # updated before being returned.

    avg_s = [unit.avg_s for unit in lay.units]
    avg_m = [unit.avg_m for unit in lay.units]
    avg_l = [unit.avg_l for unit in lay.units]
    return (avg_s, avg_m, avg_l)
end

function rel_avg_l(lay::Layer)
    # Returns the relative values of avg_l. These are the dependent
    # variables rel_avg_l in all units used in latest XCAL
    return [rel_avg_l(unit) for unit in lay.units]

end

function updt_avg_l(lay::Layer)
    # updates the long-term average (avg_l) of all the units in the
    # layer. Usually done after a plus phase.
    for i in 1:lay.N 
        updt_avg_l(lay.units[i])
    end            
end

function updt_long_avgs(lay::Layer)
    # updates the acts_p_avg and pct_act_scale variables.
    # These variables update at the end of plus phases instead of
    # cycle by cycle. 
    # This version assumes full connectivity when updating
    # pct_act_scale. If partial connectivity were to be used, this
    # should have the calculation in WtScaleSpec::SLayActScale, in
    # LeabraConSpec.cpp 
    lay.acts_p_avg = lay.acts_p_avg + lay.avg_act_dt * (acts_avg(lay) - lay.acts_p_avg);
                 
    r_avg_act_n = max(round(lay.acts_p_avg * lay.N), 1);
    lay.pct_act_scale = 1/(r_avg_act_n + 2);  
end

function reset(lay::Layer)
    # This function sets the activity of all units to random values, 
    # and all other dynamic variables are also set accordingly.            
    # Used to begin trials from a random stationary point.
    # The activity values may also be set to zero (see unit.reset)
    for i in 1:lay.N 
        reset(lay.units[i])
    end         
end

######################################################################################################
#  Network Functions
#


######################################################################################################
#  Testing
#

function test_nxx1(xp::NXX1Parameters = nxx1p)
    difTol = 1.0e-7

    tstx = [-0.05, -0.04, -0.03, -0.02, -0.01, 0, .01, .02, .03, .04, .05, .1, .2, .3, .4, .5]
    cory = [1.7735989e-14, 7.155215e-12, 2.8866178e-09, 1.1645374e-06, 0.00046864923, 0.094767615, 0.47916666, 0.65277773, 0.742268, 0.7967479, 0.8333333, 0.90909094, 0.95238096, 0.96774197, 0.9756098, 0.98039216]
    ny = Array{Float64}(undef, length(tstx))

    for i in 1:length(tstx)
        ny[i] = NoisyXX1(tstx[i])
        dif = abs(ny[i] - cory[i])
        if dif > difTol # allow for small numerical diffs
            println("XX1 err: dix: $i, x: $(tstx[i]), y: $(ny[i]), cor y: $(cory[i]), dif: $dif")
        end
    end
end

# test_nxx1()
# mylayer = layer((1,1))
# activities(mylayer)
# acts_avg(mylayer)
# scaled_acts(mylayer)
# averages(mylayer)
# rel_avg_l(mylayer)
# updt_avg_l(mylayer)
# updt_long_avgs(mylayer)
# reset(mylayer)

end # module Leabra
