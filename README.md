# Submission code for the ACM/IEEE Quantum Computing for Drug Discovery Challenge at ICCAD 2023

Structure model:

![](.assets/Layout-of-ibmq-montreal.png)

## Files

- `haml2tc.py` for constructing and saving Hamiltonian in sparse matrix format.
- `shvqe.py` for running SHVQE algorithm with the depth of cz gates as the input argument.

## Tasks

### SHVQE

Training:
- [x] Implement SHVQE.
- [ ] More seeds for initializing parameters.
- [x] Improve training settings.

To qiskit:
- [x] Hamiltonian conversion. (By Stim.)
- [x] Full run in qiskit with noise.

Results:

Clifford + 1 layer single rotation:

| depth | energy     | error (1e-2) | duration |
| ----- | ------     | ------------ | -------- |
| 3     | -78.70827  | 0.0595         | 7872     |
| 8     | -78.72387  | 0.0385         | 19392    |
| 10    | -78.730896 | 0.0291         | 24160    |
| 12    | -78.73086  | 0.0291         | -        |
| 16    | -78.74805  | 0.0060         | -        |

Only Clifford:

| depth | energy     | error  (1e-2) | duration |
| ----- | ------     | ------------- | -------- |
| **0** | -78.67825  | **0.0998**    |          |
| 8     | -78.721466 | 0.0417        | 19392    |
| 10    | -78.74937  | 0.0042        | 24160    |

Without error mitigation, depth-0 cz circuits: (see `run_shvqe_qiskit_ncz0.ipynb`)

- Duration: 320
- Shots: 1799612
- Error (%):
  - Noiseless: 0.10
  - Fake Kolkata: 1.39
  - Fake Cairo: 0.80
  - Fake Montreal: 1.65

