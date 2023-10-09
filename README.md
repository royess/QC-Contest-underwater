# Submission code for the ACM/IEEE Quantum Computing for Drug Discovery Challenge at ICCAD 2023

Team name: underwater

More detailed technical reflection is provided in `tech_reflect.pdf`.

Credit: some of codes are modified from `tensorcircuit`, `qiskit` and also the demo provided by commitee.

## Requirements

Some basic packages, `qiskit`, and `jupyter` are required.

For running the optmized circuit,
```python
pip install stim
pip install pylatexenc
pip install qiskit-aer
pip install qiskit-ignis
pip install qiskit-ibm-provider
pip install qiskit-ibm-experiment
pip install qiskit-nature
qiskit-dynamics
```

For training,
```python
pip install tensorcircuit-nightly
pip install tensorflow
```

## Instructions

The code for running optmized circuits is in `run_shvqe_qiskit_ncz0.ipynb`.

If you want to train SHVQE for yourself, please run
```bash
python shvqe_clifford.py <n>
```
where $n$ is the depth of CZ gates in Schrodinger circuit.

## Saved models

We saved the optimized circuit in `saved_models`.

- `shvqe_clifford_ncz0_sch.qasm`: optimized circuit. 
- `shvqe_clifford_ncz0_sch_transpiled.qasm`: transplied optimized circuit.

Also, we save the Heisenberg circuit as `shvqe_clifford_ncz0_hei.qasm`, which is required by the notebook `run_shvqe_qiskit_ncz0.ipynb`. Note that it is **not** real circuit but virtual circuit that is used in preprossing the Hamiltonian.
