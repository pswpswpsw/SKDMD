# EDMD/KDMD for continuous dynamical system

We implement a EDMD/KDMD module in Python for modeling continuous dynamical system.

## Steps 
1. create case by calling class object in `/source_code/classGenerateDataFromPhysics.py`
2. run EDMD/KDMD by calling class object in `/source_code/classEKDMD.py`

## Result

- the corresponding data is contained in `./result/case_name` folder
- the output model is contained in `./result/case_name` folder

## Current supported problems

- 2d polynomial case in Lusch 2017 paper
- Duffing oscillator in Otto 2017 paper

## How to add new problem?

### If you know the governing equation F(x) 
- go to `/source_code/lib` to add new models with new analytical functions

### If you only know the trajectory data but you can compute the gradient w.r.t. time
- you only need to go to `/source_code/classGenerateDataFromPhysics.py`, write a new method that 
-- 1. read the trajectory
-- 2. compute the d/dt

## Currently it only uses LatinCubic for DOE, how to incorporate advanced sampling techniques?
- go to `/source_code/classGenerateDataFromPhysics.py` line 52, change the `lhs` sampling to your method. 



## TODO
- need to have the opportuntity to disable the data generation process. otherwise running for EDMD/KDMD will generate a new data sets for each time....
