# Overview

The program created for Milestone 1 of this project solves the equations that describe
the response of a rigid membrane to an external force that acts perpendicular to the
membrane. Specifically, it solves the following partial differential equation:
```
  -omega^2 rho h  z(x,y)
    + D (d^2/dx^2 + d^2/dy^2)(d^2/dx^2 + d^2/dy^2)  z(x,y)
    - T (d^2/dx^2 + d^2/dy^2)  z(x,y)
    = P(x,y),
```
where for the moment, `P(x,y)=1`. The current version of this program solves this
equation on a square domain, but it has also been tested on circles and will, with
minor modifications to less than a dozen lines of code, be able to solve the
equation above on any mesh read from a file.

As boundary conditions, the program assumes that both the vertical deflection is
zero at the boundary, and that the plate is clamped there with a zero slope. In
other words, at the boundary the program assumes that
```
  z     = 0,
  dz/dn = 0.
```

The program solves these equations for a range of frequencies `omega` to compute
a frequency-dependent response to an external, temporally oscilating vertical 
force `P(x,y)`.

The underlying method used for the solution of the equation is the finite element
method (FEM). By default, finite element methods have difficulty dealing with
fourth order problems and historically, many complicated modifications have been
made to it for this kind of problem. Here, we instead rely on the "`C^O` Interior
Penalty" (C0IP) method by Brenner et al. The method is published in the following
article:

>   Susanne C. Brenner and Li-Yeng Sung:
>   "$C^0$ Interior Penalty Methods for Fourth Order Elliptic Boundary Value Problems on Polygonal Domains",
>   Journal of Scientific Computing, vol. 22-23, pp. 83-118, 2005.


# Input

The program reads the parameter values that determine what is to be computed from a
file called `biharmonic.prm`. This file looks as follows:
```
set Domain edge length        = 0.015
set Thickness                 = 0.0001
set Density                   = 100
set Loss angle                = 2
set Young's modulus           = 200e6
set Poisson's ratio           = 0.3
set Tension                   = 30

set Minimal frequency         = 100
set Maximal frequency         = 10000
set Number of frequency steps = 100

set Number of mesh refinement steps  = 5
set Finite element polynomial degree = 2
```
All parameters are given in SI units. `Loss angle` is dimensionless and interpreted
in degrees. The minimal and maximal frequencies are intrepreted in Hz.


# Output

The output of the program consists of two pieces, the frequency response file and the visualization
directory.

### The file `frequency_response.txt`

This file contains one line for each input
frequency omega, and in each line lists the following information:

  - The frequency, calculated as `omega/2/pi`.
  - The normalized response of the rigid membrane, as computed by the following
    equation:
    ```
      \int z(x,y) dx dy  /  \max_{x,y} |P(x,y)|.
    ```
    The output file contains both the real and imaginary component of
    this quantity. Its units are `m^3/Pa`.
  - The impedance, which is calculated as
    ```
      \max_{x,y} |P(x,y)|  /  (j  omega  \int z(x,y) dx dy).
    ```
    The output file contains both the real and imaginary component of
    this quantity. Its units are `(Pa.s)/m^3`.
  - The normalized maximal displacement of the rigid membrane, as computed by the following
    equation:
    ```
      \max_{x,y} |z(x,y)|  /  \max_{x,y} |P(x,y)|.
    ```
    This is a real-valued quantity. Its units are `m/Pa`.
  - The name of the visualization file (see below) for this frequency.


### Monitoring progress

The `frequency_response.txt` file is updated every time the program
has finished computing the response of the membrane for a particular
frequency. As a consequence, the file contains a record of all
computed frequencies.

To monitor the progress of computations -- for example for displaying
a progress bar -- open this file periodically (say, once a second) and
read what's in it. If all you want is to show progress, count the
number of lines excluding the comments at the top and divide it by the
number of frequencies provided as input. If you want something
fancier, you can actually parse the contents of the file and update a
graph of the frequency-dependent membrane response every time you read
through the file. This way, the graph will fill in over time.


### The directory `visualization/`

This directory contains one file for each input frequency, with each file providing
all of the information necessary to visualize the solution. The format used for these
files is VTU, and the solution can be visualized with either the
[Visit](https://wci.llnl.gov/simulation/computer-codes/visit) or
[Paraview](https://www.paraview.org/) programs.
The file names are of the form `visualization/solution-XXXXX.vtu` where `XXXXX`
denotes the (integer part of the) frequency (in Hz) at which the solution
is computed. However, it is easiest
to just take the file name from the `frequency_response.txt` file to find which
file corresponds to which frequency.


# Terminating execution

There may be times where callers of this program do not want it to continue with
its computations. In this case, an external program should place the text `STOP`
into a file called `termination_signal` in the current directory. This will
not immediately terminate the program; instead, it will finish the computations
for the input frequencies it is currently working on, but not start computations
for any further frequencies. Since computations on each frequency typically take
no more than a couple of seconds, this implies that the program terminates not
long after. The last step of the program is to output data for all of the
frequencies already computed into the `frequency_response.txt` file. In other
words, one has to wait for the actual program termination before reading the
results.

The program works on input frequencies in an unpredictable order, since work
for each frequency is scheduled with the operating system at the beginning
of program execution, but it is the operating system's job to allocate CPU
time for each of these tasks. This is often done in an unpredictable order.
As a consequence, the frequencies already worked on at the time of termination
may or may not be those listed first in the input file.


# How this program was tested

I have used two different ways to test the correctness of this program.
These are described below.

## Method of manufactured solutions

The "method of manufactured solutions" starts by choosing a function `Z(x,y)`
(often a combination of sines, cosines, and exponentials, but not pure
polynomials) and putting it into the equation for `z` stated at the
top of this document. For given values of the parameters `omega`, `rho`,
`h`, `D`, and `T`, one can then evaluate what pressure `P(x,y)` would be
necessary to produce this solution `Z`. This pressure `P(x,y)` is then
used as the right hand side in the program and one computes a numerical
approximation `z_h` to the solution `z=Z` of the equation for the set
of parameters used to compute `P`. It should of course
of course be close to `Z`, and furthermore, one would expect that the `z_h`
converges to `Z` as the mesh is refined, with a rate that can be inferred
from theory.

I have done this for a number of choices for `Z`, and verified both that
`z_h` is close to `Z`, and converges at the correct rate.


## Comparison with a matlab code

The second mode of comparison is with a matlab code written by Jason
McIntosh. It solves the same set of equations, but using a finite
difference method.

In the following, we show some results that compare the output of these
two programs. The initial results show that the results are largely
similar, but do have some differences. We explain these further below.

In these experiments, we use the following set of material parameters:
```
    h            = 0.000100;               // 100 microns
    rho          = 100;                    // kg/m^3
    E_angle      = 2*pi * 2./360.;         // 2 degrees
    E            = 200e6 * exp(j*E_angle); // Pa
    nu           = 0.3;                    // (Poisson's ratio)
    D            = E*h^3/12/(1-nu^2).
    T            = 30;                     // 1 N/m
```
The domain is a square with edge length `0.015m = 15mm`.


### Pure plate case

The pure plate case can be obtained by setting `T=0`. The impedances computed
by the two codes then look as follows:

![xxx](doc/images/plate/comparison.png)


### Pure membrane case

The pure plate case can be obtained by setting `D=0`. The impedances computed
by the two codes then look as follows:

![xxx](doc/images/membrane/comparison.png)

Since the only place in the equation that contains a complex-valued coefficient
is the fourth-order term with `D`, setting `D=0` results in a solution `z` that
is purely real with no imaginary part. Given the definition of the impedance,
this results in a purely imaginary impedance with no real part. The figure
reflects this.


### Combined case

Using both `D` and `T` nonzero with the values provide above, one obtains the
following comparison:

![xxx](doc/images/combined/comparison.png)


### Interpretation

The results for the two codes look qualitatively very similar, with matching
amplitudes (except at the peaks of resonances where the actual value of the
amplitude depends quite sensitively on what values of omega were used to
draw the curves) and generally also matching resonance frequencies. At the
same time, it is useful to understand where the differences may be from.

To this end, we have investigated the "pure plate" case further. For reference,
this is again the comparison between the code produced for this project and
the pre-existing matlab code:

![xxx](doc/images/plate/comparison.png)

The matlab code in its original form uses 10 grid points per wavelength. This
is generally a reasonable choice, but in this case more accuracy can be
obtained by instead using 20 grid points per wavelength. The comparison
is then as follows:

![xxx](doc/images/plate-20/comparison.png)

This minor modification illustrates that a higher mesh refinement in the
matlab code leads to a far better match of at least the lowest resonance.
On the other hand, it leaves two areas of discrepancy: The area below
1000 Hz, and the third (and probably higher) resonances.

The low-frequency discrepancy is also easily addressed: The matlab code
by default uses 10 points per wave length, but a minimum of 15 per
coordinate direction if the wave length is very long. Increasing this
minimum to 50 instead results in the following plot:

![xxx](doc/images/plate-20-50/comparison.png)

In other words, this further modification to the matlab code resolves
the discrepancy at the lowest frequencies.

Finally, the fact that now the code developed herein places the
resonance to the *right* of the matlab code (where it was to the left
before increasing the matlab mesh resolution) suggests that the
current code uses a mesh that is too coarse. Refining the mesh one
more time (i.e., halving the mesh size) then results in the following image:

![xxx](doc/images/plate-20-50-refined/comparison.png)

The resonances close to 9000 Hz are now closer together. This suggests
that an even finer mesh in the current code would lead to an even better
agreement. On the other hand, initial tests suggest that this has only a
marginal effect and further experimentation with both programs will
be necessary.

