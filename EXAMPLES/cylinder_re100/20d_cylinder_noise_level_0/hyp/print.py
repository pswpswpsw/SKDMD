#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
sys.dont_write_bytecode = True
sys.path.insert(0, '../../../../PPS_SRC')

from aprior_plot import aprior_plot

def main():
    ploter = aprior_plot('./hyp.csv')
    ploter.plot()

if __name__ == "__main__":
    main()
