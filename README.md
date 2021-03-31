# ES_adaptive_optimization
If this algorithm turns out to be useful for you, please consider citing the following work which uses the latest bounded version in a first demonstration of adaptive machine learning (AML) in which ES was combined with neural networks for automated control of electron beams in the Linac Coherent Light Source (LCLS) free electron laser (FEL) at SLAC National Accelerator Laboratory:

*Alexander Scheinker, Auralee Edelen, Dorian Bohler, Claudio Emma, and Alberto Lutman. "Demonstration of model-independent control of the longitudinal phase space of electron beams in the linac-coherent light source with femtosecond resolution." Physical review letters 121.4 (2018): 044801.*

https://doi.org/10.1103/PhysRevLett.121.044801


# Introduction and Background
Extremum seeking (ES) is a bounded, model-independent adaptive feedback algorithm for automatic tuning of 
many components simultaneously for the control and optimization of complex, nonlinear, noisy, and time-varying systems.

This method was originally designed for stabilization of coupled n-dimensional time-varying nonlinear systems of the form

<img src="https://render.githubusercontent.com/render/math?math=\dot{x}_1 = f_1(\mathbf{x},t) %2B g_1(\mathbf{x},t)u_1(\mathbf{x},t)">
<img src="https://render.githubusercontent.com/render/math?math=\vdots">
<img src="https://render.githubusercontent.com/render/math?math=\dot{x}_n = f_n(\mathbf{x},t) %2B g_n(\mathbf{x},t)u_n(\mathbf{x},t)">
with analytically unknown functions <img src="https://render.githubusercontent.com/render/math?math=f_i(\mathbf{x},t), \quad g_i(\mathbf{x},t),"> 
and user-defined feedback controls
<img src="https://render.githubusercontent.com/render/math?math=u_i(\mathbf{x},t).">

The original paper can be found here: 

*Scheinker, Alexander, and Miroslav Krstić. "Minimum-seeking for CLFs: Universal semiglobally stabilizing feedback under unknown control directions." IEEE Transactions on Automatic Control 58.5 (2012): 1107-1122.*
https://ieeexplore.ieee.org/abstract/document/6332483

In this approach, the functions <img src="https://render.githubusercontent.com/render/math?math=g_i(\mathbf{x},t)"> which multiply the control inputs 
not only change with time, but can pass through 0 changing sign, such as, for example functions of the form
<img src="https://render.githubusercontent.com/render/math?math=g_i = (1%2B2x_1-x_2)\times\cos(\omega t).">

The method has since evolved and analytical stability proofs are available for systems with non-differentiable dynamics, for 
general systems not affine in control, of the form

<img src="https://render.githubusercontent.com/render/math?math=\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x},\mathbf{u},t), \quad y(\mathbf{x},t) = h(\mathbf{x},t) %2B n(t),">

where the goal is simultaneous stabilization and minimization of a noise-corrupted (<img src="https://render.githubusercontent.com/render/math?math=n(t)">) measurement 
<img src="https://render.githubusercontent.com/render/math?math=y(\mathbf{x},t)"> of an analytically unknown time-varying function <img src="https://render.githubusercontent.com/render/math?math=h(\mathbf{x},t)">. 
Those papers can be found here:

*Scheinker, Alexander, and David Scheinker. "Bounded extremum seeking with discontinuous dithers." Automatica 69 (2016): 250-257.*
https://doi.org/10.1016/j.automatica.2016.02.023

*Scheinker, Alexander, and David Scheinker. "Constrained extremum seeking stabilization of systems not affine in control." International Journal of Robust and Nonlinear Control 28.2 (2018): 568-581.*
https://doi.org/10.1002/rnc.3886

The method has also been used for reinforcement learning (RL) applications in which an optimal feedback policy is learned online
for unknown systems with unknown objective functions purely from measurements of system dynamics, which can be found here:

*Scheinker, Alexander, and David Scheinker. "Extremum seeking for optimal control problems with unknown time‐varying systems and unknown objective functions." International Journal of Adaptive Control and Signal Processing (2020).*
https://doi.org/10.1002/acs.3097

Because this dynamic feedback for stabilization and optimization is model-independent, can tune multiple parameters simultaneously
and is robust to measurement noise, it has gained popularity in particle accelerator applications, since accelerators are usually
very large complex systems with hundreds-thousands of coupled RF and magnet components and time-varying beam distributions.
This method has now been applied to accelerators around the world including adaptive online model tuning for non-invasive electron beam
diagnostics at the Facility for Advanced Accelerator Experimental Tests (FACET) at SLAC National Accelerator Laboratory:

*Scheinker, Alexander, and Spencer Gessner. "Adaptive method for electron bunch profile prediction." Physical Review Special Topics-Accelerators and Beams 18.10 (2015): 102801.*
https://doi.org/10.1103/PhysRevSTAB.18.102801

For resonance control of RF accelerating cavities at the Los Alamos Neutron Science Center (LANSCE) linear accelerator at Los Alamos National Laboratory:

*Scheinker, Alexander. "Application of extremum seeking for time-varying systems to resonance control of RF cavities." IEEE Transactions on Control Systems Technology 25.4 (2016): 1521-1528.*
https://ieeexplore.ieee.org/abstract/document/7565566

For real-time adaptive beam trajectory control in a time-varying magnetic lattice at the SPEAR3 light source at SLAC National Accelerator Laboratory:

*Scheinker, Alexander, Xiaobiao Huang, and Juhao Wu. "Minimization of betatron oscillations of electron beam injected into a time-varying lattice via extremum seeking." IEEE Transactions on Control Systems Technology 26.1 (2017): 336-343.*
https://ieeexplore.ieee.org/abstract/document/7859370

It has been combined with machine learning (ML) in a first demonstration of adaptive machine learning (AML) for time varying systems
for the automatic control of the longitudinal phase space (LPS) of the electron beam in the Linac Coherent Light Source (LCLS) free electron laser (FEL)
at SLAC National Accelerator Laboratory:

*Alexander Scheinker, Auralee Edelen, Dorian Bohler, Claudio Emma, and Alberto Lutman. "Demonstration of model-independent control of the longitudinal phase space of electron beams in the linac-coherent light source with femtosecond resolution." Physical review letters 121.4 (2018): 044801.*
https://doi.org/10.1103/PhysRevLett.121.044801

It has been used for maximizing the output power of both the LCLS and the European XFEL FEL light sources:

*Scheinker, Alexander, et al. "Model-independent tuning for maximizing free electron laser pulse energy." Physical Review Accelerators and Beams 22.8 (2019): 082802.*
https://doi.org/10.1103/PhysRevAccelBeams.22.082802

It has been implemented for real-time online multi-objective optimization of a time-varying system at the AWAKE plasma wakefield accelerator faciltiy at CERN:

*Scheinker, Alexander, et al. "Online multi-objective particle accelerator optimization of the AWAKE electron beam line for simultaneous emittance and orbit control." AIP Advances 10.5 (2020): 055320.*
https://doi.org/10.1063/5.0003423

It has been utilized in an adaptive ML approach for automatically recovering 3D electron density maps in 3D coherent X-ray diffraction imaging:

*Scheinker, Alexander, and Reeju Pokharel. "Adaptive 3D convolutional neural network-based reconstruction method for 3D coherent diffraction imaging." Journal of Applied Physics 128.18 (2020): 184901.*
https://doi.org/10.1063/5.0014725

And this method has been utilized at the HiRES ultrafast electron diffraction (UED) beamline at Lawrence Berkeley National Laboratory:

*Scheinker, Alexander, et al. "Demonstration of adaptive machine learning-based distribution tracking on a compact accelerator: Towards enabling model-based 6D non-invasive beam diagnostics." arXiv preprint arXiv:2102.10510 (2021).*
https://arxiv.org/abs/2102.10510

# Iterative Algorithm Description
Like all adaptive methods, there are hyperparameters which must be tuned correctly for each individual application. In the attached python code, we present a
very simple example of the use of this algorithm to tune a 4-dimensional system. We hope that the following description and by playing with the code, anyone
who is interested in using this method can get an intuitive understanding of how to set hyperparameters for their specific problem of interest.

For systems of the form

<img src="https://render.githubusercontent.com/render/math?math=\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x},\mathbf{u},t), \quad y(\mathbf{x},t) = h(\mathbf{x},t) %2B n(t),">

the attached python code implements the ES algorithm by tuning parameters with controllers of the form

<img src="https://render.githubusercontent.com/render/math?math=\dot{x}_i = \sqrt{a_{ES}\omega_{i}}\cos(\omega_{i}t %2B k_{ES} \times y(\mathbf{x},t)),">

where <img src="https://render.githubusercontent.com/render/math?math=k_{ES}"> is the feedback gain, <img src="https://render.githubusercontent.com/render/math?math=a_{ES}"> controls the perturbation size for parameter tuning, and the <img src="https://render.githubusercontent.com/render/math?math=\omega_i">
are a set of distinct frequencies for the multiple parameters being tuned simultaneously. In the case of no feedback or once the system has settled near a minimum,
the parameters will oscillate about their steady-state locations <img src="https://render.githubusercontent.com/render/math?math=x_{iSS}"> with oscillations of the form

<img src="https://render.githubusercontent.com/render/math?math=x_i(t) \approx x_{iSS} %2B \sqrt{\frac{a_{ES}}{\omega_{i}}}\sin(\omega_{i}t),">

Therefore, for a given system if a particular parameter can realistically be oscillated with a dithering amplitude D, then <img src="https://render.githubusercontent.com/render/math?math=a_{ES}"> should be chosen as approximately 

<img src="https://render.githubusercontent.com/render/math?math=a_{ES} = \omega_{i}D^2.">

For digital iterative implementation, such as the most common approach in accelerators, the iterative update rule is defined based on a finite difference
approximation of the derivative:

<img src="https://render.githubusercontent.com/render/math?math=\frac{dx_i}{dt}(t) = \lim_{\Delta\rightarrow 0}\frac{x_i(t %2B \Delta)-x_i(t)}{\Delta}.">

With this approximation the ES dynamics can be rewritten as

<img src="https://render.githubusercontent.com/render/math?math=x_i(t %2B \Delta_{ES}) = x_i(t) %2B \Delta_{ES} \sqrt{a_{ES}\omega_{i}}\cos(\omega_{i}t %2B k_{ES} y(\mathbf{x},t)), \quad 0 < \Delta_{ES} \ll 1.">

For iterative applications, we drop the time argument and replace it with iterative step number n, so that the update happens according to:

<img src="https://render.githubusercontent.com/render/math?math=x_i(n %2B 1) = x_i(n) %2B \Delta_{ES} \sqrt{a_{ES}\omega_{i}}\cos(\omega_{i} n \Delta_{ES} %2B k_{ES} y(n)).">

The iterative procedure then takes place according to the following:

Step 0: Define upper and lower bounds for allowable parameter settings for all of the parameters that are being tuned. This is especially useful because these bounds can be used to normalize the parameters to all live within a common range, such as [-1,1], for various parameters with orders of magnitude differences
in their sizes such as RF phases vs magnet currents or voltages. This is done in the attached python code.

Step 1: Set initial parameter values x(1).

Step 2: Wait until the system has had time to respond and settle in response to the parameter settings.

Step 3: Record a measurement of the function of interest that is being minimized, y(1).

Step 4: Normalize all parameter values according to their bounds to within the [-1,1] space.

<img src="https://render.githubusercontent.com/render/math?math=\mathbf{x}(1) \Longrightarrow \mathbf{x}_{N}(1) \in [-1,1]^n">

Step 4: In normalized space update parameter values based on measurement y(1) according to
<img src="https://render.githubusercontent.com/render/math?math=x_{iN}(2) = x_{iN}(1) %2B \Delta_{ES} \sqrt{a_{ES}\omega_{i}}\cos(\omega_{i}\Delta_{ES} %2B k_{ES} y(1)).">

Step 5: Un-normalize the new parameter values back to their physical values.

<img src="https://render.githubusercontent.com/render/math?math=\mathbf{x}_{N}(2) \in [-1,1]^n \Longrightarrow \mathbf{x}(2) ">

Step 6: Set new parameter values x(2).

Step 7: Wait until the system has had time to respond and settle in response to the parameter changes.

Step 8: Record a new measurement of the function of interest that is being minimized, y(2).

Step 9: Normalize all parameter values again according to their bounds to within the [-1,1] space.

<img src="https://render.githubusercontent.com/render/math?math=\mathbf{x}(2) \Longrightarrow \mathbf{x}_{N}(2) \in [-1,1]^n">

Step 10: In normalized space update parameter values based on measurement y(2) according to
<img src="https://render.githubusercontent.com/render/math?math=x_{iN}(3) = x_{iN}(2) %2B \Delta_{ES} \sqrt{a_{ES}\omega_{i}}\cos(\omega_{i}2\Delta_{ES} %2B k_{ES} y(2)).">

Continue iteratively.

# Hyperparameter Tuning
In practice, for iterative tuning a good way to set the hyperparameters is as follows:

For an n-dimensional system, define the n dithering frequencies as uniformly spaced quantities in the interval [1,1.75], so that they are all distinct
and so no two frequencies are integer multiples of each other, so, for example, for a 4 dimensional system the frequencies would be:

<img src="https://render.githubusercontent.com/render/math?math=\{\omega_1, \omega_2, \omega_3, \omega_4\} = \{ 1.0, 1.25, 1.5, 1.75 \},">

next choose the ES time step as a small quantity relative to the dithering frequencies, for example, as

<img src="https://render.githubusercontent.com/render/math?math=\Delta_{ES} = \frac{2\pi}{10\times 1.75}.">

This choice ensures that the time-step is small enough so that the finite-difference approximation of ES dynamics is accurate with the highest frequency component requiring at least 10 steps to complete one full oscillation. Note that with this approach the ES time step will always be the same regardless of the number of parameters being tuned, with N parameters using the N frequencies

<img src="https://render.githubusercontent.com/render/math?math=\{\omega_1, \dots , \omega_N\} = \{ 1.0, \dots, 1.75 \}">

we can use the same ES time step

<img src="https://render.githubusercontent.com/render/math?math=\Delta_{ES} = \frac{2\pi}{10\times 1.75}.">

One way to help speed up convergence is for a system with 2N parameters to only use the N frequencies and times step:

<img src="https://render.githubusercontent.com/render/math?math=\{\omega_1, \dots , \omega_N\} = \{ 1.0, \dots, 1.75 \}, \quad \Delta_{ES} = \frac{2\pi}{10\times 1.75},">

and to alternate parameter updates between cosine and sine functions which are orthogonal in Hilbert space, so the setup might look like:

<img src="https://render.githubusercontent.com/render/math?math=x_1(n %2B 1) = x_1(n) %2B \Delta_{ES} \sqrt{a_{ES}\omega_{1}}\cos(\omega_{1} n \Delta_{ES} %2B k_{ES} y(n)),">

<img src="https://render.githubusercontent.com/render/math?math=x_2(n %2B 1) = x_2(n) %2B \Delta_{ES} \sqrt{a_{ES}\omega_{1}}\sin(\omega_{1} n \Delta_{ES} %2B k_{ES} y(n)),">

<img src="https://render.githubusercontent.com/render/math?math=x_3(n %2B 1) = x_3(n) %2B \Delta_{ES} \sqrt{a_{ES}\omega_{2}}\cos(\omega_{2} n \Delta_{ES} %2B k_{ES} y(n)),">

<img src="https://render.githubusercontent.com/render/math?math=x_4(n %2B 1) = x_4(n) %2B \Delta_{ES} \sqrt{a_{ES}\omega_{2}}\sin(\omega_{2} n \Delta_{ES} %2B k_{ES} y(n)),">

<img src="https://render.githubusercontent.com/render/math?math=\vdots">

<img src="https://render.githubusercontent.com/render/math?math=x_{2N-1}(n %2B 1) = x_{2N-1}(n) %2B \Delta_{ES} \sqrt{a_{ES}\omega_{N}}\cos(\omega_{N} n \Delta_{ES} %2B k_{ES} y(n)),">

<img src="https://render.githubusercontent.com/render/math?math=x_{2N}(n %2B 1) = x_{2N}(n) %2B \Delta_{ES} \sqrt{a_{ES}\omega_{N}}\sin(\omega_{N} n \Delta_{ES} %2B k_{ES} y(n)).">

It is also possible to try to speed up convergence by increasing the time step to something like

<img src="https://render.githubusercontent.com/render/math?math=\Delta_{ES} = \frac{2\pi}{5\times 1.75},">

but you should be careful, it can destabilize the algorithm. Usually some trial and error helps you find the limit of how far you can push it.

Once the dithering frequencies and ES time step size have been defined, I recommend starting with the ES feedback gain and amplitude both set to zero, <img src="https://render.githubusercontent.com/render/math?math=k_{ES}=0, a_{ES}=0">, and then slowly begin to increase only the dithering ampltidue <img src="https://render.githubusercontent.com/render/math?math=a_{ES}"> until you see that the parameters are making big enough changes that they are observable in the noisy measurement function. Once the perturbation size is decent, you can slowly turn up the gain <img src="https://render.githubusercontent.com/render/math?math=k_{ES}"> so that the system starts to respond.

A few images generated using the attached python code show this procedure.

First, with gain <img src="https://render.githubusercontent.com/render/math?math=k_{ES}=0"> and very small dithering amplitudes <img src="https://render.githubusercontent.com/render/math?math=a_{ES}>0"> hardly any influence on the cost function relative to the noise is visible (top) until the amplitudes are increased (bottom):
![Fig1_NoGain](https://user-images.githubusercontent.com/3331022/110808924-af45bb80-8241-11eb-9abb-a16fd23ab253.png)


Once decent oscillation sizes have been found, we slowly increase the gain to start convergence:
![Fig1_Gain](https://user-images.githubusercontent.com/3331022/110809357-0b104480-8242-11eb-82d6-fddcfe222d95.png)


Pushing the gain even further will eventuall de-stabilize the system.
![Fig1_More_Gain](https://user-images.githubusercontent.com/3331022/110809397-15324300-8242-11eb-8899-09864389294d.png)


# Automated Hyperparameter Tuning
In future work we will add algorithms that automatically tune the hyperparameters and adjust them in real time while the system is running. A few preliminary efforts towards this have shown promise, as the algorithm was able to adjust its own parameters based on analytic estimates for maximizing the light output power of the EuXFEL, as described in this work:

*Scheinker, Alexander, et al. "Model-independent tuning for maximizing free electron laser pulse energy." Physical Review Accelerators and Beams 22.8 (2019): 082802.*
https://doi.org/10.1103/PhysRevAccelBeams.22.082802
