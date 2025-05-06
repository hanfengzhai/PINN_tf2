#!/bin/bash
#SBATCH --partition=gpu-ampere
#SBATCH -n 10
#SBATCH --nodes=1
#SBATCH -G 1
#SBATCH --gres=gpu:1
#SBATCH -t 96:00:00
#SBATCH --job-name=MLPhysHW2Prob1
#SBATCH --output=MLPhysHW2Prob1.out
#SBATCH --error=MLPhysHW2Prob1.err

rm -rf figures
# Problem 1.3
python3 main_Burgers.py --actv_func 'sin' --fig_dir 'figures/Problem1_3'
python3 main_Burgers.py --actv_func 'tanh' --fig_dir 'figures/Problem1_3'

# Problem 1.4
python3 main_Burgers.py --actv_func 'sin' --iterations 10000 --fig_dir 'figures/Problem1_4'
python3 main_Burgers.py --actv_func 'tanh' --iterations 10000 --fig_dir 'figures/Problem1_4'

python3 main_Burgers.py --actv_func 'sin' --iterations 10000\
                         --fig_dir 'figures/Problem1_4_decay500' --decay 500
python3 main_Burgers.py --actv_func 'tanh' --iterations 10000\
                         --fig_dir 'figures/Problem1_4_decay500' --decay 500

# Problem 1.5
python3 main_Burgers.py --inverse_problem --fig_dir 'figures/Problem1_5'

# Problem 1.6
python3 main_Burgers.py --add_noise --fig_dir 'figures/Problem1_6'
python3 main_Burgers.py --add_noise --iterations 7500 --fig_dir 'figures/Problem1_6'