<!--
*** Thanks for checking out this README Template. If you have a suggestion that would
*** make this better, please fork the repo and create a pull request or simply open
*** an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** github_username, repo_name, twitter_handle, email
-->





<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


# SKDMD
Sparsity-promoting Kernel Dynamic Mode Decomposition for our paper on arXiv: [https://arxiv.org/abs/2002.10637](https://arxiv.org/abs/2002.10637)

## Table of Contents

* [Motivations](#motivations)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)


## Motivations

- **Why throwing everything in a neural network?**
  - neural network approach for finding Koopman operators could suffer from bad minimizers, time consuming (requires a quite large number of GPU training hours), which are all resulted from the non-convex optimization nature 

- **Why not using classical convex methods?**
  - classical nonlinear Koopman analysis method (e.g., EDMD, KDMD) suffers from having hundreds to thousands of **approximated Koopman triplets**. 
  - How to choose an **accurate** and **informative** Koopman invariant subspace in Extended/Kernel DMD?

- **Traditional linear DMD performs poorly on transient flows**
  - due to inherent linear nature, sparsity-promoting DMD (Jovanovic, et al., 2014) cannot provide accurate representation for Koopman mode decomposition in a transient regime. 
  - due to lack of truncation on nonlinearly evolving modes, **spDMD can end up with spurious unstable Koopman modes for stable flows**.

- **Not every nonlinear dynamical system requires a neural network to search for Koopman operator**
  - we resolve the issue of classical methods by rethinking modes selection in EDMD/KDMD as a *multi-task learning* problem
  - demonstration on several strongly transient flows shows the effectiveness of the algorithm for providing an accurate reduced-order-model (ROM) 

- **KDMD can be expensive to computationally evaluate (online)/train (offline)**
  - due to **cubic complexity** in the linear system step (although no one actually performs inverse exactly, but for forward computation it is inevitable)
  - we implement **random Fourier features** as a way to approximate kernel methods efficiently while enjoying the benefits of EDMD

<img src="new_framework.png" alt="drawing" width="900"/>

## Getting Started

### Prerequisites
- python3
- scipy
- numpy
- scikit-learn
- pyDOE
- mpi4py



<!-- USAGE EXAMPLES -->
## Usage

###  Example: KDMD on 2D cylinder flow past cylinder

- go to `EXAMPLE/cylinder_re100` folder

- run a standard KDMD 

  - ```python3 example_20d_cyd.py ```

- perform multi-task learning mode selection
  - note that in `run_apo_cyd_ekdmd.py`, the`max_iter = 1e5` is a **very conservative** choice for getting accurate result, one can choose `max_iter=1e2` to just get a good result of the algorithm
  - ```python3 run_apo_cyd_ekdmd.py ```
  
- postprocessing the result of multi-task learning
  - ```python3 pps_cyd.py```

- Note: that the data is top 20 POD coefficients with mean-subtracted.

#### Selection of hyperparameter

- go to `EXAMPLE/cylinder_re100/hyp` 
- run parallem hyperparameter selection 
  - ```python3 cv_kdmd_hyp_20d_cyd.py```
- then collecting the result and draw the figure by
  - ```python3 print.py```
- the resulting figure `result-num.png` is a good refernce of choosing hyperparameter


### Example: scalable EDMD on 12d flexible joint system

- go to `EXAMPLE/12d_flex_mani` folder

- run the EDMD with random Fourier features
  - ```python3 example_12d_flex_mani.py ```

- following the similar tasks in first example. 


<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/pswpswpsw/SKDMD/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Shaowu Pan - email: shawnpan@umich.edu

Project Link: [https://github.com/pswpswpsw/SKDMD](https://github.com/pswpswpsw/SKDMD)
