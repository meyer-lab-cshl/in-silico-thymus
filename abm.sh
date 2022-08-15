#!/bin/bash
#$ -cwd
#$ -l m_mem_free=5G
#$ -l h_rss=10G
#$ -l h_vmem=10G
#$ -pe threads 5

julia --threads 5 --project=. src/ThymusABM.jl
