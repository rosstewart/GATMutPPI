"""Run roc_plots.py in ablation mode (ABLATION=True, BOXPLOT=True)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import roc_plots
roc_plots.ABLATION = True
roc_plots.BOXPLOT  = True
roc_plots.main()
