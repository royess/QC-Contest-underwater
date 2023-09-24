import numpy as np
import tensorcircuit.quantum as qu
from scipy.sparse import coo_matrix, save_npz

def read_paul_string_sum_from_file(path):
    with open(path, 'r') as file:
        pauli_text_lines = file.readlines()
    l = []
    weight = []
    for line in pauli_text_lines:
        line = line.replace(' ', '')
        coeff, pauli_text_string = line.split("*")
        coeff = float(coeff)
        weight.append(coeff)
        ps = []
        for c in pauli_text_string:
            if c == 'I':
                ps.append(0)
            elif c == 'X':
                ps.append(1)
            elif c == 'Y':
                ps.append(2)
            elif c == 'Z':
                ps.append(3)
        l.append(ps)
    return l, weight

def remove_small_elements_numpy(mat):
    # Extract the values, row indices, and column indices from the COO matrix
    values = mat.data
    rows = mat.row
    cols = mat.col

    # Find the indices of elements with absolute value >= 1e-6
    mask = np.abs(values) >= 1e-6
    valid_rows = rows[mask]
    valid_cols = cols[mask]
    valid_values = values[mask]

    # Create a new COO matrix with the valid values and indices
    new_mat = coo_matrix((valid_values, (valid_rows, valid_cols)), shape=mat.shape)

    return new_mat


ham_coo = qu.PauliStringSum2COO_numpy(*read_paul_string_sum_from_file('Hamiltonian/OHhamiltonian.txt'))
ham_coo = remove_small_elements_numpy(ham_coo)
save_npz('Hamiltonian/OHham.npz', ham_coo)
