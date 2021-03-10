# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 08:53:08 2021

@author: Alexander Scheinker, ascheink@lanl.gov
"""

import numpy as np
import matplotlib.pyplot as plt

font_size = 16
plt.rcParams.update({'font.size': font_size})

#%%

# This is a very simple demonstration of how to apply the ES method
# The following example is a 4 parameter time-varying system

# Total number of ES steps to take
ES_steps = 2000

# Add Noise to the Signal
noise = np.random.randn(ES_steps)

# For the parameters being tuned, the first step is to
# define upper and lower bounds for each parameter

# Upper bounds on tuned parameters
p_max = np.array([1e6,2,400,-1e-3])

# Lower bounds on tuned parameters
p_min = np.array([5e5,-2,300,-5])

# Number of parameters being tuned
nES = len(p_min)

# Average values for normalization
p_ave = (p_max + p_min)/2.0

# Difference for normalization
p_diff = p_max - p_min

# Function that normalizes paramters
def p_normalize(p):
    p_norm = 2.0*(p-p_ave)/p_diff
    return p_norm

# Function that un-normalizes parameters
def p_un_normalize(p):
    p_un_norm = p*p_diff/2.0 + p_ave
    return p_un_norm

# Normalization allows you to easily handle a group of parameters
# which might have many orders of magnitude difference between their values
# with this normalization the normalized values live in [-1,1]

# Now we define some ES parameters

# This keeps track of the history of all of the parameters being tuned
pES = np.zeros([ES_steps,nES])

# Start with initial conditions inside of the max/min bounds
# In this case I will start them near the center of the range
pES[0] = p_ave

# This keeps track of the history of all of the normalized parameters being tuned
pES_n = np.zeros([ES_steps,nES])

# Calculate the mean value of the initial condtions
pES_n[0] = p_normalize(pES[0])

# This keeps track of the history of the measured cost function
cES = np.zeros(ES_steps)

# This is the unknown time-varying function being minimized
# For applications to a real system once parameters are set some kind of measure
# of performance is returned to the ES algorithm in place of this example function
def f_ES_minimize(p,i):
    f_val = 1e-4*np.abs(p[0]- (0.99e6-0.99e6*0.5*(1-np.exp(-i/1000))))+(p[1]-1.5)**2+2*p[2]-10.0*np.exp(-2*(p[3]+5.0)**2)
    return f_val

# Calculate the initial cost function value based on initial conditions
cES[0] = f_ES_minimize(pES[0],0) + noise[0]

# These are the unknown optimal values (just for plotting)
p_opt = np.zeros([ES_steps,nES])
p_opt[:,0] = 0.99e6 -  0.99e6*0.5*(1-np.exp(-np.arange(ES_steps)/1000))
p_opt[:,1] = 1.5 + np.zeros(ES_steps)
p_opt[:,2] = 300 + np.zeros(ES_steps)
p_opt[:,3] = -5 + np.zeros(ES_steps)


# ES dithering frequencies, for iterative applications the dithering frequencies
# are simply uniformly spread out between 1.0 and 1.75 so that no two
# frequencies are integer multiples of each other
wES = np.linspace(1.0,1.75,nES)

# ES dt step size, this particular choice of dtES ensures that the parameter
# oscillations are smooth with at least 10 steps required to complete
# one single sine() or cosine() oscillation when the gain kES = 0
dtES = 2*np.pi/(10*np.max(wES))

# ES dithering size
# In normalized space, at steady state each parameter will oscillate
# with an ampltidue of \sqrt{aES/wES}, so for example, if you'd like 
# the parameters to have normalized osciallation sizes you 
# choose the aES as:
oscillation_size = 0.1
aES = wES*(oscillation_size)**2
# Note that each parameter has its own frequency and its own oscillation size

# ES feedback gain kES (set kES<0 for maximization instead of minimization)
kES = 0.2

# The values of aES and kES will be different for each system, depending on the
# detailed shape of the functions involved, an intuitive way to set these ES
# parameters is as follows:
# Step 1: Set kES = 0 so that the parameters only oscillate about their initial points
# Step 2: Slowly increase aES until parameter oscillations are big enough to cause
# measurable changes in the noisy function that is to be minimized or maximized
# Step 3: Once the oscillation amplitudes, aES, are sufficiently big, slowly increase
# the feedback gain kES until the system starts to respond. Making kES too big
# can destabilize the system


# Decay rate. This value is optional, it causes the oscillation sizes to naturally decay.
# If you want the parameters to persistently oscillate without decay, set decay_rate = 1.0
decay_rate = 0.999

# Decay amplitude (this gets updated by the decay_rate to lower oscillation sizes
amplitude = 1.0

# This function defines one step of the ES algorithm at iteration i
def ES_step(p_n,i,cES_now,amplitude):
    
    # ES step for each parameter
    p_next = np.zeros(nES)
    
    # Loop through each parameter
    for j in np.arange(nES):
        p_next[j] = p_n[j] + amplitude*dtES*np.cos(dtES*i*wES[j]+kES*cES_now)*(aES[j]*wES[j])**0.5
    
        # For each new ES value, check that we stay within min/max constraints
        if p_next[j] < -1.0:
            p_next[j] = -1.0
        if p_next[j] > 1.0:
            p_next[j] = 1.0
            
    # Return the next value
    return p_next

# Now we start the ES loop
for i in np.arange(ES_steps-1):
    
    # Normalize previous parameter values
    pES_n[i] = p_normalize(pES[i])
    
    # Take one ES step based on previous cost value
    pES_n[i+1] = ES_step(pES_n[i],i,cES[i],amplitude)
    
    # Un-normalize to physical parameter values
    pES[i+1] = p_un_normalize(pES_n[i+1])
    
    # Calculate new cost function values based on new settings
    cES[i+1] = f_ES_minimize(pES[i+1],i+1) + noise[i+1]
    
    # Decay the amplitude
    amplitude = amplitude*decay_rate
    

    
# Plot some results
plt.figure(1,figsize=(10,15))
plt.subplot(5,1,1)
plt.title(f'$k_{{ES}}$={kES}, $a_{{ES}}$={aES}')
plt.plot(cES)
plt.ylabel('ES cost')
plt.xticks([])


plt.subplot(5,1,2)
plt.plot(pES[:,0],label='$p_{ES,1}$')
plt.plot(p_opt[:,0],'k--',label='$p_{ES,1}$ opt')
plt.legend(frameon=False)
plt.ylabel('ES parameter 1')
plt.xticks([])

plt.subplot(5,1,3)
plt.plot(pES[:,1],label='$p_{ES,2}$')
plt.plot(p_opt[:,1],'k--',label='$p_{ES,2}$ opt')
plt.legend(frameon=False)
plt.ylabel('ES parameter 2')
plt.xticks([])

plt.subplot(5,1,4)
plt.plot(pES[:,2],label='$p_{ES,3}$')
plt.plot(p_opt[:,2],'k--',label='$p_{ES,3}$ opt')
plt.legend(frameon=False)
plt.ylabel('ES parameter 3')
plt.xticks([])

plt.subplot(5,1,5)
plt.plot(pES[:,3],label='$p_{ES,4}$')
plt.plot(p_opt[:,3],'k--',label='$p_{ES,4}$ opt')
plt.legend(frameon=False)
plt.ylabel('ES parameter 4')
plt.xlabel('ES step')

plt.tight_layout()




plt.figure(2,figsize=(10,15))
plt.subplot(2,1,1)
plt.title(f'$k_{{ES}}$={kES}, $a_{{ES}}$={aES}')
plt.plot(cES)
plt.ylabel('ES cost')
plt.xticks([])


plt.subplot(2,1,2)
plt.plot(pES_n[:,0],label='$p_{ES,1,n}$')
plt.plot(pES_n[:,1],label='$p_{ES,2,n}$')
plt.plot(pES_n[:,2],label='$p_{ES,3,n}$')
plt.plot(pES_n[:,3],label='$p_{ES,4,n}$')
plt.plot(1.0+0.0*pES_n[:,0],'r--',label='bounds')
plt.plot(-1.0+0.0*pES_n[:,0],'r--')
plt.legend(frameon=False)
plt.ylabel('Normalized Parameters')
plt.xlabel('ES step')

plt.tight_layout()
















