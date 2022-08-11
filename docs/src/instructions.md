# Instructions to run ThymusABM.jl

Install Julia v1.7.2 following instructions [here](https://julialang.org/downloads/platform/)

[More useful download information](https://julialang.org/downloads/)

Additional useful information on getting started can be found [here](https://docs.julialang.org/en/v1/manual/getting-started/)

With git installed, clone the ThymusABM.jl project in a directory of your choice by typing the following in the terminal:

```
git clone https://github.com/meyer-lab-cshl/ThymusABM.jl.git
```

Login to Elzar to download a necessary `stage_genes_peptides_intlists.json` data file. This can be done by typing the following in the terminal, and then entering your password:

```
ssh username@bamdev1
```

Type the following in the terminal to navigate to the directory containing the file:

```
cd /grid/meyer/home/mulle
```

Type the following to copy the data file to the data directory in the ThymusABM project directory (replace username@localmachine with your username and address on your local machine and replace /path/to/ with the preceding path to the ThymusABM directory):

```
scp stage_genes_peptides_intlists.json username@localmachine:/path/to/ThymusABM/data
```

Navigate to the ThymusABM project directory on your local machine and type julia into the terminal

With Julia running, press ] to enter pkg mode and type:

```
(v1.7) pkg> activate .

(ThymusABM) pkg> instantiate
```

To run, navigate to the ThymusABM project directory and type in the terminal: 

```
julia --threads 1 --project=. src/ThymusABM.jl
```

This will take approximately 5 minutes to run, with the current step being printed out after every 100 steps. Upon successful simulation completion, the following message will be printed: "Simulation complete. Data written to ThymusABM/data folder as adf4600thymocytes7threshold10synapsecomplexes1rngseed.csv and mdf4600thymocytes7threshold10synapsecomplexes1rngseed.csv" along with the time taken to run.

There should now be an adf4600thymocytes7threshold10synapsecomplexes1rngseed.csv and a mdf4600thymocytes7threshold10synapsecomplexes1rngseed.csv file in the ThymusABM/data/results folder that holds collected data for agent-related parameters and model-related parameters, respectively. The first few lines of adf.csv should be as follows:

| step | count_tec | count_check_stage_1_tec | count_check_stage_2_tec | count_check_stage_3_tec | count_check_stage_4_tec | count_check_stage_5_tec | count_check_stage_6_tec | count_check_stage_7_tec | count_check_stage_8_tec | count_check_stage_9_tec |
|------|-----------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|
| 0    | 956       | 83                      | 117                     | 105                     | 105                     | 108                     | 127                     | 117                     | 88                      | 106                     |
| 1    | 956       | 101                     | 91                      | 112                     | 105                     | 108                     | 112                     | 124                     | 113                     | 90                      |
| 2    | 956       | 104                     | 88                      | 109                     | 109                     | 106                     | 113                     | 120                     | 118                     | 89                      |
| 3    | 956       | 103                     | 89                      | 110                     | 108                     | 107                     | 113                     | 120                     | 117                     | 89                      |
| 4    | 956       | 104                     | 89                      | 108                     | 111                     | 104                     | 113                     | 122                     | 116                     | 89                      |

The first few lines of mdf.csv should be as follows:

| step | num_tregs | autoreactive_thymocytes | escaped_thymocytes | nonautoreactive_thymocytes   | alive_thymocytes | escape_ratio | react_ratio       | nonreact_ratio     | threshold | total_thy | alive_ratio      | escapedautoreactive_ratio | model_volume |
|------|-----------|-------------------------|--------------------|------------------------------|------------------|--------------|-------------------|--------------------|-----------|-----------|------------------|---------------------------|--------------|
| 0    | 0         | 0                       | 0                  | 0                            | 4600             | NaN          | NaN               | NaN                | 7         | 4600      | Inf              | NaN                       | 0.001        |     
| 1    | 0         | 0                       | 0                  | 40                           | 4600             | 0            | 0                 | 1                  | 7         | 4640      | 115              | NaN                       | 0.001        |
| 2    | 0         | 1                       | 0                  | 80                           | 4600             | 0            | 0.012345679012345 | 0.987654320987654  | 7         | 4681      | 56.7901234567901 | 0                         | 0.001        |
| 3    | 0         | 3                       | 0                  | 108                          | 4600             | 0            | 0.027027027027027 | 0.972972972972973  | 7         | 4711      | 41.4414414414414 | 0                         | 0.001        |
| 4    | 0         | 5                       | 0                  | 128                          | 4600             | 0            | 0.037593984962406 | 0.962406015037594  | 7         | 4733      | 34.5864661654135 | 0                         | 0.001        |

The parameters that can be configured in config.toml for the ABM are listed below.

```
steps
    How long to run the ABM for
    Type: Int
    Example: 1000
volume
    Volume of ABM environment (mm^3)
    Type: Float64
    Example: 0.001
num_medullas
    Number of medullas to simulate (amount of simulations to run at once)
    Type: Int
    Example: 5
selection_threshold
    Threshold that a thymocyte's binding strength to a peptide needs to match or exceed to be negatively selected
    Type: Int
    Example: 7
treg_threshold
    Threshold for a thymocyte to be classified as a Treg after it successfully exits the thymus without being negatively selected
    Type: Int
    Example: 6
thymocyte_speed
    Spped that the thymocytes will travel at
    Type: Int16
    Example: 0.015
rng
    Array of random number generator seeds to use (the length of this array should be equal to num_medullas)
    Type: Array of Ints
    Example: [1, 2, 3, 4, 5]
synapse_interactions
    Amount of reactions to calculate for one interaction between APC and thymocyte
    Type: Int
    Example: 10
minimum_strength
    Minimum strength needed in one pMHC:TCR binding to ensure the peptide is remembered by the thymocyte
    Type: Int
    Example: 2
matrixtype
    Type of matrix to use to calculate binding strengths
    Type: String
    Example: "hamming"
```
