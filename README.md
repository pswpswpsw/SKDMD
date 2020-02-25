# SKDMD
Sparsity-promoting Kernel Dynamic Mode Decomposition

## Motivations

- **Why throwing everything in a neural network?**
  - neural network approach for finding Koopman operators suffered from bad minimizer, time consuming, resulting from the non-convex optimization nature 
- **Why not using classical convex algorithm?**
  - classical nonlinear Koopman analysis method (e.g., EDMD, KDMD) suffers from choosing an **accurate** and **informative** Koopman invariant subspace in Kernel DMD
- **Not every nonlinear dynamical system requires a neural network to search for Koopman operator**
  - we resolve the issue of classical traditional by rethinking modes selection in EDMD/KDMD as a *multi-task learning* problem
  - demonstration on several strongly transient flows shows the effectiveness of the algorithm for providing an accurate reduced-order-model (ROM) 


<img src="new_framework.png" alt="drawing" width="900"/>



# Requirement
- python3
- scipy
- numpy
- scikit-learn
- pyDOE
- mpi4py

# 2D cylinder flow past cylinder

- go to `EXAMPLE/cylinder_re100` folder

- run a standard KDMD 

  - ```python3 example_20d_cyd.py ```

- perform multi-task learning mode selection
  - note that in `run_apo_cyd_ekdmd.py`, the`max_iter = 1e5` iis a safe choice for getting accurate result, one can choose `max_iter=1e3` to just get a try of the algorithm
  - ```python3 run_apo_cyd_ekdmd.py ```
  
- postprocessing the result of multi-task learning
  - ```python3 pps_cyd.py```

# Selection of hyperparameter

- go to `EXAMPLE/cylinder_re100/hyp` 
- run parallem hyperparameter selection 
  - ```python3 cv_kdmd_hyp_20d_cyd.py```
- then collecting the result and draw the figure by
  - ```python3 print.py```
- the resulting figure `result-num.png` is a good refernce of choosing hyperparameter

