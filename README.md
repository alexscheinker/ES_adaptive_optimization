# ES_adaptive_optimization
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

https://ieeexplore.ieee.org/abstract/document/6332483

In this approach, the functions <img src="https://render.githubusercontent.com/render/math?math=g_i(\mathbf{x},t)"> which multiply the control inputs 
not onyl change with time, but can pass through 0 changing sign, such as, for example functions of the form
<img src="https://render.githubusercontent.com/render/math?math=g_i = (1%2B2x_1-x_2)\times\cos(\omega t).">

The method has since evolved and analytical stability proofs are available for systems with non-differentiable dynamics, for 
general systems not affine in control, of the form

<img src="https://render.githubusercontent.com/render/math?math=\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x},\mathbf{u},t), \quad y = h(\mathbf{x},t) %2B n(t),">

where the goal is simultaneous stabilization and minimization of a noise-corrupted (<img src="https://render.githubusercontent.com/render/math?math=n(t)">) measurement 
<img src="https://render.githubusercontent.com/render/math?math=y(\mathbf{x},t)"> of an analytically unknown time-varying function <img src="https://render.githubusercontent.com/render/math?math=h(\mathbf{x},t)">. 
Those papers can be found here:

https://doi.org/10.1016/j.automatica.2016.02.023

https://doi.org/10.1002/rnc.3886

The method has also been used for reinforcement learning applications in which an optimal feedback policy is learned online
for unknown systems with unknown objective functions purely from measurements of system dynamics, which can be found here:

https://doi.org/10.1002/acs.3097

Because this dynamic feedback for stabilization and optimization is model-independent, can tune multiple parameters simultaneously
and is robust to measurement noise, it has gained popularity in particle accelerator applications, since accelerators are usually
very large compelx systems with hundreds-thousands of coupled RF and magnet components and time-varying beam distributions.
This method has now been applied to accelerators around the world including adaptive online model tuning for non-invasive electron beam
diagnostics at the Facility for Advanced Accelerator Experimental Tests (FACET) at SLAC National Accelerator Laboratory:

https://doi.org/10.1103/PhysRevSTAB.18.102801

for resonance control of RF accelerating cavities at the Los Alamos Neutron Science Center (LANSCE) linear accelerator at Los Alamos National Laboratory:

https://ieeexplore.ieee.org/abstract/document/7565566

for real-time adaptive beam trajectory control in a time-varying magnetic lattice at the SPEAR3 light source at SLAC National Accelerator Laboratory:

https://ieeexplore.ieee.org/abstract/document/7859370

it has been combined with machine learning (ML) in a first demonstration of adaptive machine learning (AML) for time varying systems
for the automatic control of the longitudinal phase space (LPS) of the electron beam in the Linac Coherent Light Source (LCLS) free electron laser (FEL)
at SLAC National Accelerator Laboratory:

https://doi.org/10.1103/PhysRevLett.121.044801

it has been used for maximizing the output power of both the LCLS and the European XFEL FEL light sources:

https://doi.org/10.1103/PhysRevAccelBeams.22.082802

has been implemented for real-time online multi-objective optimization of a time-varying system at the AWAKE plasma wakefield accelerator faciltiy at CERN:

https://doi.org/10.1063/5.0003423

has been utilized in an adaptive ML approach for automatically recovering 3D electron density maps in 3D coherent X-ray diffraction imaging:

https://doi.org/10.1063/5.0014725

and has been utilized at the HiRES ultrafast electron diffraction (UED) beamline at Lawrence Berkeley National Laboratory:

https://arxiv.org/abs/2102.10510



