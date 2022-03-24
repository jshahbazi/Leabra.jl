module Leabra

######################################################################################################
# Imports
#

using Distributions
using Random
using LinearAlgebra

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

# g = conductance
#       ge = excitatory conductance - net excitatory input to the neuron
#       gi = inhibitory conductance - net inhibitory input to the neuron
# E = driving potential
#       Ee = excitatory driving potential
# V = voltage potential
#       Vm = membrane potential
#       Vmr = reset potential = v_reset
# 0 = action potential threshold = thr = spk_thr
# I = current
# dtvm = rate constant that determines how fast membrance potential changes
#       dtvm = vm_dt = ac.Dt.VmDt = v_dt

# Summary of Neuron Equations and Normalized Parameters
# Table 2.1 From CCN 4th Ed.
# Parameter     Bio Val     Norm Val
# Time	        0.001 sec	1ms 
# Current	    1x10−8 A 	10 nA 
# Capacitance   1x10−12 F 	1pF 
# GbarL (leak)  10 nS	    0.1
# GbarE (excite)100 nS 	    1
# ErevI (inhib)	-75mV 	    0.25
# θ (Thr)	     -50mV	    0.5
# Voltage 	    100mV	    0..2
# Conductance 	1x10−9 S	1 nS
# C (memb cap) 	281 pF	    Dt = .355
# GBarI (inhib) 100 nS	    1
# ErevL (leak) 	-70mV	    0.3
# ErevE (excite)0mV	        1
# SpikeThr	    20mV	    1.2


mutable struct Unit # rate code approximation
    act::Float64        # = 0.2           # "firing rate" of the unit
    avg_ss::Float64     # = act           # super-short time-scale activation average
    avg_s::Float64      # = act           # short time-scale activation average
    avg_m::Float64      # = act           # medium time-scale activation average
    avg_l::Float64      # = 0.1           # long time-scale average of medium-time scale (trial level) activation, used for the BCM-style floating threshold in XCAL  
    g_e::Float64        # = 0.0           # net input. Asymptotically approaches g_e_raw (see cycle).
    v_m::Float64        # = 0.3           # membrane potential
    vm_eq::Float64      # = 0.3           # a version of v_m that doesn't reset with spikes
    adapt::Float64      # = 0.0           # adaptation, as in the AdEx model 
    spike::Bool         # = false         # a flag that indicates spiking threshold was crossed

    # constants
    # const g_bar_e::Float64     # = 0.3
    # const g_bar_l::Float64     # = 0.3
    const g_e_dt::Float64     # = 1/1.4   # time step constant for update of 'g_e'
    const integ_dt::Float64   # = 1.0     # time step constant for integration of cycle dynamics
    const vm_dt::Float64      # = 1/3.3   # time step constant for membrane potential
    const l_dn_dt::Float64    # = 1/2.5   # time step constant for avg_l decrease   # TODO never used
    const adapt_dt::Float64   # = 1/144   # time step constant for adaptation
    const ss_dt::Float64      # = 0.5     # time step for super-short average
    const s_dt::Float64       # = 0.5     # time step for short average
    const m_dt::Float64       # = 0.1     # time step for medium-term average
    const avg_l_dt::Float64   # = 0.1     # time step for long-term average
    const avg_l_max::Float64  # = 1.5     # max value of avg_l
    const avg_l_min::Float64  # = 0.1     # min value of avg_l
    const avg_l_gain::Float64 # = 2.5     
    
    const e_rev_e::Float64    # = 1.0     # excitatory reversal potential
    const e_rev_i::Float64    # = 0.25    # inhibitory reversal potential
    const e_rev_l::Float64    # = 0.3     # leak reversal potential
    const g_bar_l::Float64    # = 0.1     # leak conductance
    const g_bar_e::Float64    # = 1.0     # excitatory conductance
    const thr::Float64        # = 0.5     # normalized "rate threshold"
    const spk_thr::Float64    # = 1.2     # normalized spike threshold
    const vm_r::Float64       # = 0.3     # reset potential after spike
    const vm_gain::Float64    # = 0.04    # gain that voltage produces on adaptation
    const spike_gain::Float64 # = 0.00805 # effect of spikes on adaptation
    const l_up_inc::Float64   # = 0.2     # increase in avg_l if avg_m has been 'large'   # TODO never used  # maybe: (increase in avg_l if avg_m has been "large")
    
    function Unit()
        return new( 0.2, 0.2, 0.2, 0.2, 0.1, 0.0, 0.3, 0.3, 0.0, false,
                    1/1.4, 1.0, 1/3.3, 1/2.5, 1/144, 0.5, 0.5, 0.1, 0.1, 1.5, 0.1, 2.5, 1.0, 0.25, 0.3, 0.1, 1.0, 0.5, 1.2, 0.3, 0.04, 0.00805, 0.2)
     end    
end
    
function rel_avg_l(u::Unit)::Float64
    return (u.avg_l - u.avg_l_min)/(u.avg_l_max - u.avg_l_min)
end

function cycle(u::Unit, g_e_raw::Float64, g_i::Float64)
    # Does one Leabra cycle. Called by the layer cycle method.
    # g_e_raw = instantaneous, scaled, received input
    # g_i = fffb inhibition
    
    ## updating net input
    # Ge +=      DtParams.Integ * (1/ DtParams.GTau) * (GeRaw - Ge)
    u.g_e = u.g_e + u.integ_dt * u.g_e_dt * (g_e_raw - u.g_e)
    
    ## Finding membrane potential
    #Inet = Ge *    (Erev.E  - Vm)    + Gbar.L * (Erev.L - Vm)    +    Gi * (Erev.I - Vm) + Noise
    i_e = u.g_e     * (u.e_rev_e - u.v_m)
    i_l = u.g_bar_l * (u.e_rev_l - u.v_m)    
    i_i =   g_i     * (u.e_rev_i - u.v_m)
    i_net = i_e + i_l + i_i #+ rand() # noise?
    
    # almost half-step method for updating v_m (adapt doesn't half step)
    v_m_half = u.v_m + 0.5 * u.integ_dt * u.vm_dt * (i_net - u.adapt)
    i_e_h    = u.g_e     * (u.e_rev_e - v_m_half)
    i_l_h    = u.g_bar_l * (u.e_rev_l - v_m_half)    
    i_i_h    =   g_i     * (u.e_rev_i - v_m_half)
    i_net_h  = i_e_h + i_l_h + i_i_h
    u.v_m    = u.v_m + u.integ_dt * u.vm_dt * (i_net_h - u.adapt)

    # new rate coded version of i_net
    i_e_r = u.g_e     * (u.e_rev_e - u.vm_eq)
    i_l_r = u.g_bar_l * (u.e_rev_l - u.vm_eq)    
    i_i_r =   g_i     * (u.e_rev_i - u.vm_eq)
    i_net_r = i_e_r + i_l_r + i_i_r
    u.vm_eq = u.vm_eq + u.integ_dt * u.vm_dt * (i_net_r - u.adapt)    
    
    # finding whether there's an action potential
    if u.v_m > u.spk_thr
        u.spike = true
        u.v_m = u.vm_r
        i_net = 0.0
    else
        u.spike = false
    end

    # finding instantaneous rate due to input
    # if Act < XX1Params.VmActThr && Vm <= X11Params.Thr: 
    if u.act < nxx1p.VmActThr && u.vm_eq <= u.thr
        # nwAct = NoisyXX1(Vm - Thr)
        nw_act = nxx1(u.vm_eq - u.thr)[1]
    else
        ## Finding activation
        # finding threshold excitatory conductance
        # geThr = (Gi * (Erev.I -  Thr)   + Gbar.L * (Erev.L - Thr) / (Thr - Erev.E)
        g_e_thr = (g_i * (u.e_rev_i - u.thr) + u.g_bar_l * (u.e_rev_l - u.thr) - u.adapt) / (u.thr - u.e_rev_e)
        # nwAct = NoisyXX1(Ge * Gbar.E - geThr)
        nw_act = nxx1(u.g_e * u.g_bar_e - g_e_thr)[1]
    end

    # update activity
    # y(t) = y(t − 1) + dtvm (y∗(x) − y(t − 1))
    # Act += (1 / DTParams.VmTau) * (nwAct - Act)
    # vm_dt is 1/vm_tau
    u.act = u.act + u.integ_dt * u.vm_dt * (nw_act - u.act)


    ## Updating adaptation current
    u.adapt = u.adapt + u.integ_dt * (u.adapt_dt * (u.vm_gain * (u.v_m - u.e_rev_l)  - u.adapt) + u.spike * u.spike_gain)
          
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
    u.act = input
    
    ## updating averages
    u.avg_ss = u.avg_ss + u.integ_dt * u.ss_dt * (u.act    - u.avg_ss)
    u.avg_s  =  u.avg_s + u.integ_dt * u.s_dt  * (u.avg_ss - u.avg_s)
    u.avg_m  =  u.avg_m + u.integ_dt * u.m_dt  * (u.avg_s  - u.avg_m)
end


function updt_avg_l(u::Unit)
    # This fuction updates the long-term average 'avg_l' 

    # AvgL += (1 / Tau) * (Gain * AvgM - AvgL); AvgL = MAX(AvgL, Min)
    # Tau = 10, Gain = 2.5 (this is a key param -- best value can be lower or higher) Min = .2
    u.avg_l = u.avg_l + u.avg_l_dt * (u.avg_l_gain * u.avg_m - u.avg_l)
    u.avg_l = max(u.avg_l, 0.2) #note does the 0.2 nullify u.avg_l_min?
    
    # matlab code
    # if u.avg_m > 0.2
    #     u.avg_l = u.avg_l + u.avg_l_dt*(u.avg_l_max - u.avg_m)
    # else
    #     u.avg_l = u.avg_l + u.avg_l_dt*(u.avg_l_min - u.avg_m)
    # end
end

function reset(u::Unit, random::Bool=false)
    # This function sets the activity to a random value, and sets
    # all activity time averages equal to that value.
    # Used to begin trials from a random stationary point.
    if random
        u.act = rand(Uniform(0.05,0.95))
    else
        u.act = 0.0
    end
    u.avg_ss = u.act
    u.avg_s = u.act
    u.avg_m = u.act
    u.avg_l = u.act
    u.g_e = 0.0
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
        results[index] = NoisyXX1(value, xp)
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
    ge_avg::Float64
    ge_max::Float64
    maxvsage::Float64  # def:"0,0.5,1" desc:"what proportion of the maximum vs. average netinput to use in the feedforward inhibition computation -- 0 = all average, 1 = all max, and values in between = proportional mix between average and max (ff_netin = avg + ff_max_vs_avg * (max - avg)) -- including more max can be beneficial especially in situations where the average can vary significantly but the activity should not -- max is more robust in many situations but less flexible and sensitive to the overall distribution -- max is better for cases more closely approximating single or strictly fixed winner-take-all behavior -- 0.5 is a good compromise in many cases and generally requires a reduction of .1 or slightly more (up to .3-.5) from the gi value for 0"`
    wt::Array{Float64, 2}
    ce_wt::Array{Float64, 2}
    N::Int64
    fbi::Float64

    const fb_dt::Float64        # = 1/1.4 # Integration constant for feedback inhibition (fb_tau=1.4)
    const ff::Float64           # = 1.0   # feedforward scaling of inhibition
    const ff0::Float64          # = 0.1   # threshold
    const fb::Float64           # = 0.5   # feedback scaling of inhibition    
    const gi::Float64           # = 1.8   # inhibition multiplier
    const avg_act_dt::Float64   # = 0.01
end

Base.show(io::IO, lay::Layer) = print(io, 
                                      "Layer - ", length(lay.units), " Units\n",
                                      "pct_act_scale: ",lay.pct_act_scale,"\n",
                                      "acts_p_avg: ",   lay.acts_p_avg,"\n",
                                      "ge_avg: ",    lay.ge_avg,"\n",
                                      "wt: ",           lay.wt,"\n",
                                      "ce_wt: ",        lay.ce_wt,"\n",
                                      "N: ",            lay.N,"\n",
                                      "fbi: ",          lay.fbi,"\n",
                                      "ff: ",           lay.ff,"\n",
                                      "ff0: ",          lay.ff0,"\n",
                                      "fb: ",           lay.fb,"\n",
                                      "fb_dt: ",        lay.fb_dt,"\n",
                                      "gi: ",           lay.gi,"\n",
                                      "avg_act_dt: ",   lay.avg_act_dt,"\n",
                                     )

function layer(dims::Tuple{Int64, Int64} = (1,1))
    N = dims[1]*dims[2]

    units = [Unit() for i in 1:N]
    
    acts_avg = 0.2
    avg_act_n =  1.0
    pct_act_scale = 1/(avg_act_n + 2)
    fb = 0.5

    lay = Layer(units, 
                pct_act_scale, 
                acts_avg, 
                0.0, # ge_avg
                0.0, # maxvsavg
                0.0, # ge_max
                zeros(Float64, (1,1)),
                zeros(Float64, (1,1)),
                N,
                fb * acts_avg,
                1/1.4, 1.0, 0.1, fb, 1.8, 0.01
    )

    return lay
end

function acts_avg(lay::Layer)
    return mean(activities(lay))
end

function activities(lay::Layer)
    # returns a vector with the activities of all units
    acts = zeros(Float64, lay.N)
    for (index,unit) in enumerate(lay.units)
        acts[index] = unit.act
    end
    return transpose(acts)
end

function scaled_acts(lay::Layer)
    # returns a vector with the scaled activities of all units
    acts = Array{Float64, 1}(undef, lay.N)
    for (index,unit) in enumerate(lay.units)
        acts[index] = unit.act
    end
    return lay.pct_act_scale .* acts
end

function cycle(lay::Layer, raw_inputs::Array{Float64}, ext_inputs::Array{Float64})
    # this function performs one Leabra cycle for the layer
    #raw_inputs = A vector of size I, where I is the total number of inputs
    #             from all layers. Each input has already been scaled
    #             by the pct_act_scale of its layer of origin and by
    #             the wt_scale_rel factor.
    #ext_inputs = A vector of size N denoting inputs that don't come
    #             from another layer, where N is the number of
    #             units in this layer. An empty vector indicates
    #             that there are no external inputs.
     
    ## obtaining the net inputs    
    # GeRaw += Sum_(recv) Prjn.GScale * Send.Act * Wt        
    ge_raw = lay.ce_wt * raw_inputs  # contrast-enhanced weights are used
    if any(ext_inputs) .> 0.0
        ge_raw = ge_raw .+ ext_inputs
    end

    ## obtaining inhibition
    lay.ge_avg = mean(ge_raw)

    # it seems for a majority of the time, this won't be done, but
    # it's here just in case
    if lay.maxvsage > 0.0
        for value in ge_raw
            lay.ge_max = 0.0 # TODO does this go here?
            if value > lay.ge_max
                lay.ge_max = value
            end
        end
        # ffNetin = avgGe + FFFBParams.MaxVsAvg * (maxGe - avgGe)
        ffNetin = lay.ge_avg + lay.maxvsage * (lay.ge_max - lay.ge_avg)
    else
        ffNetin = lay.ge_avg
    end
    

    # ffi = FFFBParams.FF * MAX(ffNetin - FFBParams.FF0, 0)
    ffi = lay.ff * max(ffNetin - lay.ff0, 0)

    # fbi += (1 / FFFBParams.FBTau) * (FFFBParams.FB * avgAct - fbi
    lay.fbi = lay.fbi + lay.fb_dt * (lay.fb * acts_avg(lay) - lay.fbi)
    # Gi = FFFBParams.Gi * (ffi + fbi)
    g_i = lay.gi * (ffi + lay.fbi)
    
    ## calling the cycle method for all units
    for i in 1:lay.N
        cycle(lay.units[i], ge_raw[i], g_i)
    end
end

function clamped_cycle(lay::Layer, input::Array{Float64})
    # sets all unit activities equal to the input and updates all
    # the variables as in the cycle function.
    # input = vector specifying the activities of all units
    for i = 1:lay.N  # parfor ?
        clamped_cycle(lay.units[i], input[i])
    end
    lay.fbi = lay.fb * acts_avg(lay)
end
        
function averages(lay::Layer)
    # Returns the s,m,l averages in the layer as vectors.
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
    lay.acts_p_avg = lay.acts_p_avg + lay.avg_act_dt * (acts_avg(lay) - lay.acts_p_avg)
                 
    r_avg_act_n = max(round(lay.acts_p_avg * lay.N), 1)
    lay.pct_act_scale = 1/(r_avg_act_n + 2)  
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

mutable struct Network
    layers::Array{Layer}
    connections::Array{Float64, 2}
    n_lays::Int64
    n_units::Int64
    lrate::Float64
    
    # TODO add type of layer for avg_l_lrn, it is 0.0004 on default, but 0 for output layers (they do not need self-organized learning).
    const avg_l_lrn_max::Float64    # = 0.01    # max amount of "BCM" learning in XCAL
    const avg_l_lrn_min::Float64    # = 0.0     # min amount of "BCM" learning in XCAL
    const lrn_m::Float64            # = 0.1     # proportion of medium to short term avgs. in XCAL
    const m_lrn::Float64            # = 1       # proportion of error-driven learning in XCAL
    const d_thr::Float64            # = 0.0001  # threshold for XCAL "check mark" function
    const d_rev::Float64            # = 0.1     # reversal value for XCAL "check mark" function
    const off::Float64              # = 1.0     # 'offset' in the SIG function for contrast enhancement
    const gain::Float64             # = 6.0     # gain in the SIG function for contrast enhancement
end

function network(dim_layers, connections, w0)
    # constructor to the network class.
    # dim_layers = 1-D cell array. dim_layers{i} is a vector [j,k], 
    #              where j is the number of rows in layer i, and k is
    #              the number of columns in layer i.
    # connections = a 2D array. If layer j sends projections to 
    #               layer i, then connections[i,j] = c > 0; 0 otherwise
    # w0 = w0 is a cell array. w0{i,j} is the weight matrix with the
    #      initial weights for the connections from layer j to i             
    
    n_lay = length(dim_layers)  # number of layers
    (nrc, ncc) = size(connections)

    @assert nrc == ncc "Non-square connections matrix in network constructor"
    @assert nrc == n_lay "Number of layers inconsistent with connection matrix in network constructor"
    @assert sum(size(w0) .== size(connections)) >= 2 "Inconsistent dimensions between initial weights and connectivity specification in network constructor"
    @assert all(connections .>= 0) "Negative projection strengths between layers are not allowed in connections matrix"
   
    net = Network(Array{Layer, 1}(undef, n_lay),
                  zeros(Float64, (n_lay, n_lay)),
                  n_lay,
                  0,   # counting number of units in the network
                  0.1, # set default learning rate
                  0.01, 0.0, 0.1, 1, 0.0001, 0.1, 1.0, 6.0 #constants
    )

    ## Normalizing the rows of 'connections' so they add to 1
    row_sums = sum(connections, dims=2)
    for row in 1:n_lay
        if row_sums[row] > 0
            connections[row,:] = connections[row,:] ./ row_sums[row]
        end
    end

    for i in 1:n_lay
        net.layers[i] = layer(dim_layers[i])
        net.n_units = net.n_units + net.layers[i].N
    end              

    ## Second test of argument dimensions
    for i in 1:n_lay
        for j in 1:n_lay
            if w0[i,j] == 0.0
                if connections[i,j] > 0.0
                    throw("Connected layers have no connection weights")
                end
            else
                if connections[i,j] == 0.0
                    throw("Non-empty weight matrix for unconnected layers")
                end

                (r,c) = size(w0[i,j])
                if net.layers[i].N != r || net.layers[j].N != c
                    throw("Initial weigths are inconsistent with layer dimensions")
                end
            end
        end
    end
    net.connections = connections # assigning layer connection matrix
    
    ## Setting the inital weights for each layer
    # first find how many units project to the layer in all the network
    lay_inp_size::Array{Int64} = zeros(1,n_lay)
    for i in 1:n_lay
        for j in 1:n_lay
            if connections[i,j] > 0  # if layer j projects to layer i
                lay_inp_size[i] = lay_inp_size[i] + net.layers[j].N
            end
        end
    end

    # add the weights for each entry in w0 as a group of columns in wt
    for i in 1:n_lay
        net.layers[i].wt = zeros(Float64, (net.layers[i].N, lay_inp_size[i]))
        index = 1
        for j = 1:n_lay
            if connections[i,j] > 0  # if layer j projects to layer i
                nj = net.layers[j].N
                net.layers[i].wt[:,index:index+nj-1] = w0[i,j]
                index = index + nj
            end
        end
    end 
    
    # set the contrast-enhanced version of the weights
    for lay in 1:n_lay
        # TODO check book version of contrast-enhanced weights: 1 / (1 + (self$weights) / (off * (1 - self$weights)) ^ -gain)
        net.layers[lay].ce_wt = 1 ./ (1 .+ (net.off * (1 .- net.layers[lay].wt) ./ net.layers[lay].wt) .^ net.gain)
    end

    return net
end

function XCAL_learn(net::Network)
    # XCAL_learn() applies the XCAL learning equations in order to
    # modify the weights in the network. This is typically done at
    # the end of a plus phase. The equations used come from:
    # https://grey.colorado.edu/ccnlab/index.php/Leabra_Hog_Prob_Fix#Adaptive_Contrast_Impl
    # Soft weight bounding and contrast enhancememnt are as in:
    # https://grey.colorado.edu/emergent/index.php/Leabra
    
    ## updating the long-term averages
    for lay in 1:net.n_lays
        updt_avg_l(net.layers[lay]) 
    end
    
    ## Extracting the averages for all layers            
    avg_s = Array{Vector{Float64}, 1}(undef, net.n_lays)
    avg_m = Array{Vector{Float64}, 1}(undef, net.n_lays)
    avg_l = Array{Vector{Float64}, 1}(undef, net.n_lays)
    avg_s_eff = Array{Vector{Float64}, 1}(undef, net.n_lays)

    for lay in 1:net.n_lays
        (avg_s[lay], avg_m[lay], avg_l[lay]) = averages(net.layers[lay]) # layer.averages() function
        # AvgSLrn = (1-LrnM) * AvgS + LrnM * AvgM
        avg_s_eff[lay] = net.lrn_m * avg_m[lay] + (1 - net.lrn_m) * avg_s[lay]
    end
    
    ## obtaining avg_l_lrn
    avg_l_lrn = Array{Vector{Float64}, 1}(undef, net.n_lays)
    for lay in 1:net.n_lays
        # AvgLLrn = ((Max - Min) / (Gain - Min)) * (AvgL - Min)
        avg_l_lrn[lay] = net.avg_l_lrn_min .+ rel_avg_l(net.layers[lay]) .* (net.avg_l_lrn_max - net.avg_l_lrn_min)
    end
    
    ## For each connection matrix, calculate the intermediate vars.
    srs = Array{Array{Float64, 2}, 2}(undef, (net.n_lays,net.n_lays)) # srs{i,j} = matrix of short-term averages
                                                                        # where the rows correspond to the
                                                                        # units of the receiving layer, columns
                                                                        # to units of the sending layer.
    srm = Array{Array{Float64, 2}, 2}(undef, (net.n_lays,net.n_lays)) # ditto for avg_m
    for rcv in 1:net.n_lays
        for snd in rcv:net.n_lays
            # notice we only calculate the 'upper triangle' of the
            # cell arrays because of symmetry
            if net.connections[rcv,snd] > 0 || net.connections[snd,rcv] > 0
                # srs = Send.AvgSLrn * Recv.AvgSLrn
                srs[rcv,snd] = avg_s_eff[rcv] * transpose(avg_s_eff[snd])
                # srm = Send.AvgM * Recv.AvgM
                srm[rcv,snd] = avg_m[rcv] * transpose(avg_m[snd])

                if snd != rcv # using symmetry
                    srs[snd,rcv] = transpose(srs[rcv,snd])
                    srm[snd,rcv] = transpose(srm[rcv,snd])                             
                end
            end
        end
    end
    
    ## calculate the weight changes
    dwt = Array{Array{Float64, 2}, 2}(undef, (net.n_lays,net.n_lays)) # dwt{i,j} is the matrix of weight changes
                                                                      # for the weights from layer j to i
    for rcv in 1:net.n_lays
        for snd in 1:net.n_lays
            if net.connections[rcv,snd] > 0
                sndN = net.layers[snd].N
                # dwt = XCAL(srs, srm) + Recv.AvgLLrn * XCAL(srs, Recv.AvgL)
                temp_dwt = ( net.m_lrn .* xcal(net, srs[rcv,snd], srm[rcv,snd]) .+ ((expand(avg_l_lrn[rcv], 2, size(srs[rcv,snd])[2])) .* xcal(net, srs[rcv,snd], transpose(repeat(transpose(avg_l[rcv]), sndN)))))
                # DWt = Lrate * dwt
                dwt[rcv,snd] = net.lrate .* temp_dwt
            end
        end
    end

    ## update weights (with weight bounding)
    for rcv in 1:net.n_lays
        # DWt = 0
        DW = zeros(Float64, (net.n_lays,net.n_lays))
        isempty_DW = true
        for snd in 1:net.n_lays
            if net.connections[rcv,snd] > 0
                if isempty_DW
                    DW = dwt[rcv,snd]
                else
                    DW = hcat(DW, dwt[rcv,snd])
                end
                isempty_DW = false
            end
        end
        if !isempty_DW
            # Here's the weight bounding part, as in the CCN book
            # DWt *= (DWt > 0) ? Wb.Inc * (1-LWt) : Wb.Dec * LWt
            # LWt += DWt
            idxp = net.layers[rcv].wt .> 0
            idxn = .!idxp # maps the "!" function onto the BitArray
            net.layers[rcv].wt[idxp] = net.layers[rcv].wt[idxp] .+ (1 .- net.layers[rcv].wt[idxp]) .* DW[idxp]
            net.layers[rcv].wt[idxn] = net.layers[rcv].wt[idxn] .+ net.layers[rcv].wt[idxn] .* DW[idxn]
        end
    end
    
    ## set the contrast-enhanced version of the weights
    for lay in 1:net.n_lays
        # Wt = SIG(LWt)
        # SIG(w) = 1 / (1 + (Off * (1-w)/w)^Gain)
        net.layers[lay].ce_wt = 1 ./ (1 .+ (net.off .* (1 .- net.layers[lay].wt) ./ net.layers[lay].wt) .^ net.gain)
    end
end

function updt_long_avgs(net::Network)
    # updates the acts_p_avg and pct_act_scale variables for all layers. 
    # These variables update at the end of plus phases instead of
    # cycle by cycle. The avg_l values are not updated here.
    # This version assumes full connectivity when updating
    # pct_act_scale. If partial connectivity were to be used, this
    # should have the calculation in WtScaleSpec::SLayActScale, in
    # LeabraConSpec.cpp 
    for lay in 1:net.n_lays
        updt_long_avgs(net.layers[lay])
    end
end

function m1(net::Network)
    # obtains the m1 factor: the slope of the left-hand line in the
    # "check mark" XCAL function. Notice it includes the negative
    # sign.
    return (net.d_rev - 1.0) / net.d_rev
end
        
function xcal(net::Network, x, th)
    @assert size(x) == size(th)
    # this function implements the "check mark" function in XCAL.
    # x = an array of abscissa values.
    # th = an array of threshold values, same size as x

    # XCAL(x, th) = (x < DThr) ? 0 : (x > th * DRev) ? (x - th) : (-x * ((1-DRev)/DRev))
    # DThr = 0.0001, DRev = 0.1 defaults, and x ? y : z terminology is C syntax for: if x is true, then y, else z

    # python approach
    # if x < net.d_thr
    #     return 0
    # elseif x > (net.d_rev * th)
    #     return (x - th)
    # else
    #     return (-x * ((1 - net.d_rev)/net.d_rev))
    # end


    f = zeros(size(x))
    temp = x .> net.d_thr
    temp2 = x .< (net.d_rev * th)
    idx1 = temp .& temp2
    idx2 = x .>= (net.d_rev * th)

    f[idx1] = m1(net) * x[idx1]
    f[idx2] = x[idx2] - th[idx2]
    return f      
end

function reset(net::Network)
    # This function sets the activity of all units to random values, 
    # and all other dynamic variables are also set accordingly.            
    # Used to begin trials from a random stationary point.
    for lay = 1:net.n_lays
        reset(net.layers[lay])
    end
end

function set_weights(net::Network, w::Array{Matrix{Float64}, 2})
    # This function receives a cell array w, which is like the cell
    # array w0 in the constructor: w{i,j} is the weight matrix with
    # the initial weights for the connections from layer j to layer
    # i. The weights are set to the values of w.
    # This whole function is a slightly modified copypasta of the
    # constructor.
    
    ## First we test the dimensions of w
    @assert sum(size(w) == size(net.connections)) >= 2 "Inconsistent dimensions between weights and connectivity specification in set_weights"

    for i in 1:net.n_lays
        for j in 1:net.n_lays
            if all(w[i,j] .== 0.0) # if isempty(w[i,j])
                if net.connections[i,j] > 0
                    throw("Connected layers have no connection weights")
                end
            else
                if net.connections[i,j] == 0
                    throw("Non-empty weight matrix for unconnected layers")
                end
                (r,c) = size(w[i,j])
                if net.layers[i].N != r || net.layers[j].N != c
                    throw("Initial weights are inconsistent with layer dimensions")
                end
            end
        end
    end
    
    ## Now we set the weights
    # first find how many units project to the layer in all the network
    lay_inp_size = zeros(1,net.n_lays)
    for i in 1:net.n_lays
        for j in 1:net.n_lays
            if net.connections[i,j] > 0  # if layer j projects to layer i
                lay_inp_size[i] = lay_inp_size[i] + net.layers[j].N
            end
        end
    end
    
    # add the weights for each entry in w0 as a group of columns in wt
    for i in 1:net.n_lays
        net.layers[i].wt = zeros(net.layers[i].N,lay_inp_size[i])
        index = 1
        for j in 1:net.n_lays
            if net.connections[i,j] > 0  # if layer j projects to layer i
                nj = net.layers[j].N
                net.layers[i].wt[:,index:index+nj-1] = w[i,j]
                index = index + nj
            end
        end
    end   
    
    # set the contrast-enhanced version of the weights
    for lay in 1:net.n_lays
        net.layers[lay].ce_wt = 1 ./ (1 .+ (net.off .* (1 .- net.layers[lay].wt) ./ net.layers[lay].wt) .^ net.gain)
    end
end

function get_weights(netnet::Network)
    # This function returns a 2D cell array w.
    # w{rcv,snd} contains the weight matrix for the projections from
    # layer snd to layer rcv.
    w = Array{Array{Float64, 2}, 2}(undef, (net.n_lays,net.n_lays))
    for rcv in 1:net.n_lays
        idx1 = 1 # column where the weights from layer 'snd' start
        for snd in 1:net.n_lays
            if net.connections[rcv,snd] > 0
                Nsnd = net.layers[rcv].N
                w[rcv,snd] = net.layers[rcv].wt[:,idx1:idx1+Nsnd-1]
                idx1 = idx1 + Nsnd
            end
        end               
    end
    return w
end

function cycle(net::Network, inputs::Vector{Array{Float64}}, clamp_inp::Bool)
    # this function calls the cycle method for all layers.
    # inputs = a cell array. inputs{i} is  a matrix that for layer i
    #          specifies the external input to each of its units.
    #          An empty matrix denotes no input to that layer.
    # clamp_inp = a binary flag. 1 -> layers are clamped to their
    #             input value. 0 -> inputs summed to ge_raw.
    
    ## Testing the arguments and reshaping the input
    @assert length(inputs) == net.n_lays "Number of layers inconsistent with number of inputs in network cycle"

    for inp in 1:net.n_lays  # reshaping inputs into column vectors
        if any(inputs[inp] .> 0.0) #if ~isempty(inputs[inp])
            inputs[inp] = reshape(inputs[inp], net.layers[inp].N, 1)
        end
    end
    
    ## First we set all clamped layers to their input values
    clamped_lays = zeros(1, net.n_lays)
    if clamp_inp
        for lay in 1:net.n_lays
            if any(inputs[lay] .> 0.0) # if ~isempty(inputs[lay])
                clamped_cycle(net.layers[lay], inputs[lay])
                clamped_lays[lay] = 1
            end
        end
    end
    
    ## We make a copy of the scaled activity for all layers
    # scaled_acts = zeros(Float64, 1, net.n_lays)
    scaled_acts_array = Array{Vector{Float64}}(undef, net.n_lays)
    for lay in 1:net.n_lays
        scaled_acts_array[lay] = scaled_acts(net.layers[lay])
    end
    
    ## For each unclamped layer, we put all its scaled inputs in one
    #  column vector, and call its cycle function with that vector
    for recv in 1:net.n_lays
        if all(clamped_lays[recv] .== 0.0) # if the layer is not clamped
            # for each 'recv' layer we find its input vector
            long_input = zeros(1, net.n_units) # preallocating for speed
            n_inps = 0  # will have the # of input units to 'recv'
            n_sends = 0 # will have the # of layers sending to 'recv'
            conns = (net.connections[recv, :] .> 0.0)
            wt_scale_rel = net.connections[recv, conns]
            for send in 1:net.n_lays
                if net.connections[recv, send] > 0
                    n_sends = n_sends + 1
                    long_input[(1 + n_inps):(n_inps + net.layers[send].N)] = wt_scale_rel[n_sends] .* scaled_acts_array[send]
                    n_inps = n_inps + net.layers[send].N
                end
            end
            # now we call 'cycle'
            cycle(net.layers[recv], long_input[1:n_inps], inputs[recv])                   
        end
    end
    
end




######################################################################################################
# Higher level functions for building networks
#

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

function clamp_data(data, num_layers, layers_to_clamp, selected_data)
    resulting_data = Vector{Array{Float64}}(undef, num_layers)
    output_layer::Bool = false

    for layer in 1:num_layers
        if layer == num_layers
            output_layer = true
        end

        if layer in layers_to_clamp
            if output_layer
                resulting_data[layer] = data[selected_data,2]
            else
                resulting_data[layer] = data[selected_data,1]
            end
        else
            resulting_data[layer] = []
        end
    end

    return resulting_data
end

function train_network!(net, inputs, n_epochs, n_trials)
    n_minus = 50 # number of minus cycles per trial
    n_plus = 25  # number of plus cycles per trial

    lrate_sched = collect(LinRange(0.8, 0.2, n_epochs))
    errors = zeros(n_epochs,n_trials)
    for epoch in 1:n_epochs
        randomized_input = randperm(n_trials)
        net.lrate = lrate_sched[epoch]
        for trial in 1:n_trials
            Leabra.reset(net)
            selected_data = randomized_input[trial] # pick a random set of data

            # Minus phase
            # Clamp input layer
            sel_data_array = clamp_data(inputs, net.n_lays, [1], selected_data)
            for minus in 1:n_minus
                Leabra.cycle(net, sel_data_array, true)
            end
            outs = (Leabra.activities(net.layers[net.n_lays]))

            # Plus phase
            # Clamp input and output layer
            sel_data_array = clamp_data(inputs, net.n_lays, [1,net.n_lays], selected_data)
            for plus in 1:n_plus
                Leabra.cycle(net, sel_data_array, true);
            end
            Leabra.updt_long_avgs(net)

            # Learning
            Leabra.XCAL_learn(net)
            
            # Errors
            errors[epoch, selected_data] = 1 - sum(outs .* transpose(inputs[selected_data, 2][:])) / ( norm(outs) * norm(transpose(inputs[selected_data, 2][:])) );                    
        end
    end

    return errors
end




######################################################################################################
# Hacky Export All Function 
#

for n in names(@__MODULE__; all=true)
    if Base.isidentifier(n) && n ∉ (Symbol(@__MODULE__), :eval, :include)
        @eval export $n
    end
end

end # module Leabra
