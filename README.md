## Running this code

### Virtual environment
I recommend using a virtual environment to manage dependencies and execute python code.

venv and conda are both good options:
https://docs.python.org/3/library/venv.html
https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html

### Installing depedencies
All dependencies for this code can be found in the requirements.txt file and can be installed with:
pip install -r requirements.txt


### Running code

`python runQRC.py --num_gates=10 --gate_set=G3 --observables_type=all --err_type=thermal_relaxation --t1=50e6 --t2=0.01`

where

+ `num_gates` is the number of gates (results presented in paper used values: 25,50 ..)
+ `gates_set` is the name of the gate set, from the list [G1, G2, G3, MG, D2, D3, Dn]
+ `observable_type` is either *single* which only returns the expected values, *fidelity* which returns the state fidelities or *all*, which returns the expected values, fidelitites and final states
+ `error_type` must be *thermal_relaxation*, *amplitude_damping*, *depolarizing*, *phase_damping* or *none*
+ `t1` is the thermal relaxation time (typical value is 50e-6)
+ `t2` is the thermal dephasing time (typical value is 70e-6)



## Contact  

Feel free to contact me to discuss any issues, questions or comments.

* GitHub: [matthewjball](https://github.com/matthewjball)

### BibTex reference format for citation for the Code
```
@misc{QRCgithubmball,
title={The advantage of noise in quantum reservoir computing},
url={https://github.com/matthewjball/QRC-noise-simulations},
note={GitHub repository containing code for simulations of different sources of noise in qiskit on molecular data. Forked from and built on work hosted here: https://github.com/laiadc/Optimal_QRC_noise},
author={Matthew Ball},
  year={2026}
}
```

