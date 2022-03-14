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

struct NXX1Params
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

    function Unit()
        return new(0.5, 100, 0.005, 0.01, 0.33, 0.8, 3.0, 0.01, 10.0, 0.1, 
                   3.0/0.005, (0.33 * ((100*0.005)^0.8)), 0.5 * (0.33 * ((100*0.005)^0.8)), 
                   XX1GainCor(0.01,10.0,0.005,100,0.1) - (0.5 * (0.33 * ((100*0.005)^0.8))) )
    end
end


function XX1GainCor(x::Float64, xp::NXX1Params)
    gainCorFact = (xp.GainCorRange - (x / xp.NVar)) / xp.GainCorRange
    if gainCorFact < 0
        return XX1(xp.Gain * x)
    end
    newGain = xp.Gain * (1 - xp.GainCor*gainCorFact)
    return XX1(newGain * x)
end  

function NoisyXX1(x::Float64, xp::NXX1Params)
	if x < 0
		return xp.SigMultEff / (1 + exp(-(x * xp.SigGainNVar)))
	elseif x < xp.InterpRange
		interp = 1 - ((xp.InterpRange - x) / xp.InterpRange)
		return xp.SigValAt0 + interp*xp.InterpVal
	else
		return XX1GainCor(x, xp)
    end
end

function XX1GainCorGain(x::Float64, gain::Float64, xp::NXX1Params)
	gainCorFact = (xp.GainCorRange - (x / xp.NVar)) / xp.GainCorRange
	if gainCorFact < 0
		return XX1(gain * x)
    end
	newGain = gain * (1 - xp.GainCor*gainCorFact)
	return XX1(newGain * x)
end

function NoisyXX1Gain(x::Float64, gain::Float64, xp::NXX1Params)
	if x < xp.InterpRange
		sigMultEffArg = xp.SigMult * (gain*xp.NVar ^ xp.SigMultPow)
		sigValAt0Arg = 0.5 * sigMultEffArg

		if x < 0 # sigmoidal for < 0
			return sigMultEffArg / (1 + exp(-(x * xp.SigGainNVar)))
		else # else x < interp_range
			interp = 1 - ((xp.InterpRange - x) / xp.InterpRange)
			return sigValAt0Arg + interp*xp.InterpVal
        end
	else
		return xp.XX1GainCorGain(x, gain, xp)
    end
end

function nxx1(points)
    xp = NXX1Params()
    results = Array{Float64}(undef, length(points))
    for (index,value) in enumerate(points)
        results[index] = NoisyXX1(value, xp)
    end
    return results
end




end # module Leabra
