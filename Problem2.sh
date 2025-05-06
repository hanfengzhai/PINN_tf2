#!/bin/bash
#SBATCH --partition=gpu-ampere
#SBATCH -n 10
#SBATCH --nodes=1
#SBATCH -G 1
#SBATCH --gres=gpu:1
#SBATCH -t 96:00:00
#SBATCH --job-name=MLPhysHW2Prob2
#SBATCH --output=MLPhysHW2Prob2.out
#SBATCH --error=MLPhysHW2Prob2.err

# Problem 2.2
python3 main_ODE.py --option 'Prob1_2' --fig_dir 'figures/Problem2_2'

# Problem 2.3
python3 main_ODE.py --option 'Prob3' --fig_dir 'figures/Problem2_3'
python3 main_ODE.py --option 'Prob3' --fig_dir 'figures/Problem2_3_gam' --gamma 0.1
python3 main_ODE.py --option 'Prob3' --fig_dir 'figures/Problem2_3_gam' --gamma 100
python3 main_ODE.py --option 'Prob3' --fig_dir 'figures/Problem2_3_exp' --actv_func 'exp'

