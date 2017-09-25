# MPEM
Matrix Product Equation of Motion: Matrix Product State approximation for dynamical processes on networks.

If you use this code, please cite:

- [1] Thomas Barthel, Caterina De Bacco and Silvio Franz, *A matrix product algorithm for stochastic dynamics on locally tree-like graphs*, 	arXiv:1508.03295, (2015).

Paper preprint available [here](http://arxiv.org/abs/1508.03295).

Copyright (c) 2017 Caterina De Bacco

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## What's included:
- `python` : Python code and a test script as follows.
- `python/one_time_update.py`. Contains the main algorithmic steps for the matrix product state decomposition.
- `python/tools_dynamics.py`. Contains two examples of dynamics transition probabilities: the majority rule and Glauber dynamics. Feel free to add new types of rules.
- `python/tools_observables.py`. Contains functions to calculate observables like marginals and magnetization.
- `python/tools_main.py`. Contains command line parameters sepcifications.
- `data` : Contains sample result for the dynamics of some observables.

## Notes:
Need to make a directory called `data` at the same level of the `python` folder. 
To make one, just type from the command line, inside that folder: 
* `mkdir data`

For a usage example, inside folder `python/` type in the command line:
* `python test_mpem.py`

Feel free to modify the script `python test_mpem.py` or to add command line parameters.
The dynamics transition probabilities are specified in `python/tools_dynamics.py`. Feel free to add new types of dynamics.

#### Input format.
If you use the script `test_mpem.py` then you can specify various parameter from the command line:
- `h` or --help gives you the parameters explanation
- `d` sets graph connected component: 0=Keep the whole graph; 1=Keep only the max connected component
- `n` sets graph number of nodes
- `k` sets average degree
- `t` sets T_max, max iteration time
- `s` sets svd_routine: 0=fix the number of singular values; 1=fix the norm ratio between truncated and exact matrices. 
- `p` sets the svd_parameter: either the max number of singular values or the norm cutoff ratio
- `i` sets random number generator seed
- `m` sets max number of singular values accepted
- `J` sets mean value Js  (needed for Glauber Spin-Glass dynamics)
- `j` sets variance Js    (needed for Glauber Spin-Glass dynamics)
- `b` sets initial bias spin      
- `B` sets beta            (needed for Glauber Spin-Glass dynamics)
- `e` sets type of dynamic 0=majority; 1=Glauber   

#### Output.
Various files will be generated inside the `data` folder, see the script `python/test_mpem.py` for files names and quantities in output. Feel free to modify the script to output what you prefer.



