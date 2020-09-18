#!/bin/sh
python3 Plot_CUR_results.py
python3 Plot_PCA_results.py
python3 Plot_FPS_results.py
python3 Plot_RR_results.py
jupyter nbconvert --to notebook --execute --inplace Plot_Figures_3-4-5.ipynb
