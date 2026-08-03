## Forked work, credit to original author
This repository was forked from previous work investigating the effect of noise on quantum reservoir computing
Code hosted here: https://github.com/laiadc/Optimal_QRC_noise/blob/main/README.md
Findings presented here: https://www.nature.com/articles/s41598-023-35461-5

The summary of my contributions to this forked repository are:
- updating API calls for qiskit version <2.0
- Adding noise model for thermal noise
- Updating simulation code to use density matrix model
- Circuit simulation now done via GPU
- Various performance optimisations
- Notebook(s) updated to generate figures for thermal noise model

Full credit goes to the original author for the existing work including:
- Generating molecular data
- QRC algorithms
- Qubit and gate preparation
- Logic for collection of observables results
- Notebooks for statistical analysis and generating figures

This project was used as part of my master's thesis on the topic of investigating the effects of noise on quantum computation. In particular this work is interested in exploring whether noise can be exploited for computational advantages.

Simulation parameters were set to be realistic based on available hardware and for fair comparison with the results previously obtained in https://www.nature.com/articles/s41598-023-35461-5.

## Running this code

### Virtual environment
I recommend using a virtual environment to manage dependencies and execute python code.

venv and conda are both good options:
https://docs.python.org/3/library/venv.html
https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html

### Installing depedencies
All dependencies for this code can be found in the requirements.txt file and can be installed with:
`pip install -r requirements.txt`

Note that qiskit-aer-gpu is only available on x86_64 Linux distributions and requires Nvidia CUDA >=11 to operate.

If your system does not meet these requirements instead use the qiskit-aer library and remove parameter 'device="GPU"' from simulator objects.

More information here: https://pypi.org/project/qiskit-aer-gpu/


### Running code

Simulation code is contained within the runQRC.py file and requires the following parameters:

+ `num_gates` is the number of gates (results presented in paper used values: 25,50 ..)
+ `gates_set` is the name of the gate set, from the list [G1, G2, G3, MG, D2, D3, Dn]
+ `observable_type` is either *single* which only returns the expected values, *fidelity* which returns the state fidelities or *all*, which returns the expected values, fidelitites and final states
+ `error_type` must be *thermal_relaxation*, *amplitude_damping*, *depolarizing*, *phase_damping* or *none*
+ `t1` is the thermal relaxation time (typical value is 50e-6)
+ `t2` is the thermal dephasing time (typical value is 70e-6)

Example command:
`python runQRC.py --num_gates=10 --gate_set=G3 --observables_type=all --err_type=thermal_relaxation --t1=50e6 --t2=0.01`

Additionally this repo contains a script 'run_therm_sims.sh' that I used to orchestrate simulations with different parameters.


## Contact  

Feel free to contact me to discuss any issues, questions or comments.

* GitHub: [matthewjball](https://github.com/matthewjball)

### BibTex reference format for citation for the Code
```
@misc{QRCgithubmball,
title={QRC noise simulations},
url={https://github.com/matthewjball/QRC-thermal-noise-simulation},
note={GitHub repository containing code for simulations of different sources of noise in qiskit on molecular data. Forked from and built on work hosted here: https://github.com/laiadc/Optimal_QRC_noise},
author={Matthew Ball},
  year={2026}
}
```

