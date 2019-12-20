# Overview

This is a program to solve the equations that describe the response of a rigid membrane to
an external force that acts perpendicular to the membrane. Specifically, it solves the
following partial differential equation:
```
  -omega^2 rho h  z(x,y)
    + D (d^2/dx^2 + d^2/dy^2)(d^2/dx^2 + d^2/dy^2)  z(x,y)
    - T (d^2/dx^2 + d^2/dy^2)  z(x,y)
    = P(x,y)
```
where for the moment, `P(x,y)=1`. This equation is solved on a circle of given radius.

...need to still change this to read from a domain provided by upstream...


# Input


# Output 

The output of the program consists of two pieces, the frequency response file and the visualization
directory.

### The file `frequency_response.txt`

This file contains one line for each input
frequency omega, and in each line lists the following information:
  - The frequency omega
  - The normalized response of the rigid membrane, as computed by the following
    equation:
      \int z(x,y) dx dy  /  \int P(x,y) dx dy
  - The normalized maximal displacement of the rigid membrane, as computed by the following
    equation:
      \max_{x,y} |z(x,y)|  /  \max_{x,y} |P(x,y)|
  - The name of the visualization file (see below) for this frequency.

### The directory `visualization/`

This directory contains one file for each input frequency, with each file providing
all of the information necessary to visualize the solution. The format used for these
files is VTU, and the solution can be visualized with either the
[Visit](https://wci.llnl.gov/simulation/computer-codes/visit) or
[Paraview](https://www.paraview.org/) programs.
The file names are of the form `visualization/solution-XXXX.XXXXX.vtu` where `XXXX.XXXXX`
denotes the frequency at which the solution is computed. However, it is easiest
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
