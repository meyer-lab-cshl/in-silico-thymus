# Instructions
Install Julia vX.X following instructions here: https://julialang.org/downloads/

Additional useful information on getting started can be found here: https://docs.julialang.org/en/v1/manual/getting-started/

Simply clone their project using e.g. git clone, cd to the project directory and call
```
(v1.0) pkg> activate .

(SomeProject) pkg> instantiate
```

## Example

Below is an example of how to run the ABM using Julia, assuming it and all required packages are installed:
```
julia ThymusABM.jl --steps 2000 --n_tecs 500 --n_dendritics 50 --n_thymocytes 5000 --selection_threshold 2.5 --dt 1.0 --rng 42 --synapse_interactions 3 --min_strong_interactions 1 --dim1 10.0 --dim2 10.0 --dim3 10.0
```

The parameters that can be specified when running the ABM are listed below. If any parameter is not specified, the default value will be used.

```
--steps
    How long to run the ABM for
    arg_type = Int
    default = 1000
--dim1
    Length of 1st dimension
    arg_type = Float64
    default = 10.0
--dim2
    Length of 2nd dimension
    arg_type = Float64
    default = 10.0
--dim3
    Length of 3rd dimension
    arg_type = Float64
    default = 10.0
--n_tecs
    Number of mTECs to place in ABM
    arg_type = Int
    default = 500
--n_dendritics
    Number of dendritic cells to place in ABM
    arg_type = Int
    default = 50
--n_thymocytes
    Number of thymocytes to place in ABM
    arg_type = Int
    default = 5000
--selection_threshold
    Reaction threshold needing to be met to cause negative selection
    arg_type = Float64
    default = 1.6
--dt
    Size of time step - important for velocity/position calculations
    arg_type = Float64
    default = 1.0
--rng
    Random number generator seed to use
    arg_type = Int
    default = 1
--synapse_interactions
    Amount of reactions to calculate for one interaction between APC and thymocyte
    arg_type = Int
    default = 1
--min_strong_interactions
    Minimum amount of strong reactions needed in one APC-thymocyte interaction to induce negative selection
    arg_type = Int
    default = 1
```
