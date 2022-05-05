# Instructions to run ThymusABM.jl

Install Julia v1.7.2 following instructions [here](https://julialang.org/downloads/platform/)

[More useful download information](https://julialang.org/downloads/)

Additional useful information on getting started can be found [here](https://docs.julialang.org/en/v1/manual/getting-started/)

With git installed, clone the ThymusABM.jl project in a directory of your choice by typing the following in the terminal:

```
git clone https://github.com/meyer-lab-cshl/ThymusABM.jl.git
```

Login to Elzar to download a necessary data file. This can be done by typing the following in the terminal, and then entering your password:

```
ssh username@bamdev1
```

Type the following in the terminal to navigate to the directory containing the file:

```
cd /grid/meyer/home/mulle
```

Type the following to copy the data file to the data directory in the ThymusABM project directory (replace /path/to/ with the preceding path to the ThymusABM directory):

```
scp matches.dist username@localmachine:/path/to/ThymusABM/data
```

Navigate to the ThymusABM project directory on your local machine and type julia into terminal

With Julia running, press ] to enter pkg mode and type:

```
(v1.7) pkg> activate .

(ThymusABM) pkg> instantiate
```

To run, navigate to ThymusABM project directory and type in the terminal: 

```
julia --project=. src/ThymusABM.jl --steps 2000 --n_tecs 500 --n_dendritics 50 --n_thymocytes 5000 --selection_threshold 250 --dt 1.0 --rng 42 --synapse_interactions 3 --min_strong_interactions 1 --dim1 10.0 --dim2 10.0 --dim3 10.0
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
    arg_type = Int16
    default = 100
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
