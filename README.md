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
- [ ] Add process bar.

To qiskit:
- [x] Hamiltonian conversion. (By Stim.)
- [ ] Full run in qiskit with noise.

Results:

Clifford + 1 layer single rotation:

| depth | energy     | error | duration |
| ----- | ------     | ----- | -------- |
| 3     | -78.70827  | 5.95  | 7872     |
| 8     | -78.72387  | 3.85  | 19392    |
| 10    | -78.730896 | 2.91  | 24160    |
| 12    | -78.73086  | 2.91  | -        |
| 16    | -78.74805  | 0.60  | -        |

Only Clifford:

| depth | energy     | error | duration |
| ----- | ------     | ----- | -------- |
| 8     | -78.721466 | 4.17  | 19392    |
| 10    | -78.74937  | 0.42  | 24160    |
