import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.providers.aer.noise import NoiseModel
from qiskit_aer import AerSimulator
import numpy as np
import pandas as pd
from mitiq import Executor, MeasurementResult, Observable
from cirq import DensityMatrixSimulator, depolarize
from mitiq.interface import convert_to_mitiq
import mitiq
from mitiq import zne, benchmarks, pec
from mitiq.pec.representations import find_optimal_representation
from mitiq.pec.representations.depolarizing import represent_operations_in_circuit_with_local_depolarizing_noise
from mitiq.pec.channels import kraus_to_super
from qiskit.circuit.library import RZGate
import pdb


def standard_gate_unitary(name):
    # To be removed with from_dict
    unitary_matrices = {
        ("id", "I"): np.eye(2, dtype=complex),
        ("x", "X"): np.array([[0, 1], [1, 0]], dtype=complex),
        ("y", "Y"): np.array([[0, -1j], [1j, 0]], dtype=complex),
        ("z", "Z"): np.array([[1, 0], [0, -1]], dtype=complex),
        ("h", "H"): np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
        ("s", "S"): np.array([[1, 0], [0, 1j]], dtype=complex),
        ("sx", "SX"): np.array([[1+1j, 1-1j],[1-1j, 1+1j]], dtype=complex) / 2,
        ("sdg", "Sdg"): np.array([[1, 0], [0, -1j]], dtype=complex),
        ("t", "T"): np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
        ("tdg", "Tdg"): np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex),
        ("cx", "CX", "cx_01"): np.array(
            [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex
        ),
        ("cx_10",): np.array(
            [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex
        ),
        ("cz", "CZ"): np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex
        ),
        ("swap", "SWAP"): np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
        ),
        ("ccx", "CCX", "ccx_012", "ccx_102"): np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=complex,
        ),
        ("ccx_021", "ccx_201"): np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
            ],
            dtype=complex,
        ),
        ("ccx_120", "ccx_210"): np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ],
            dtype=complex,
        ),
    }

    return next((value for key, value in unitary_matrices.items() if name in key), None)



noise_dict = pd.read_pickle('./NoiseModel/fakecairo.pkl')

single_qubit_noise_operator = []
# noise_model = NoiseModel.from_dict(noise_dict)
# for name in noise_model.noise_instructions:
    
for noise in noise_dict['errors']:
    if len(noise['gate_qubits'][0])==1 and noise['gate_qubits'][0][0]==0:
        single_qubit_noise_operator.append(noise)


# we first need to define basis circuits supporting our noise model, for each element of basis_circuits describes "how to physically implement" a noisy operation

noisy_super_op_list = []

for noisy_operator in single_qubit_noise_operator:
    if noisy_operator['operations'][0]=='' or noisy_operator['operations'][0]=='measure':
        continue
    
    ideal_operation_kraus = []
    for i in range(len(noisy_operator['operations'])):
        kraus_sets = []
        print(noisy_operator['operations'][i])
        if noisy_operator['operations'][i]=='reset':
            ideal_operation_kraus.append(np.array([[1,0],[0,0]],dtype=np.complex128))
            ideal_operation_kraus.append(np.array([[0,1],[0,0]],dtype=np.complex128))

        else:
            kraus_operator = standard_gate_unitary(noisy_operator['operations'][i])
            ideal_operation_kraus.append(kraus_operator)
    super_op_list = []
    prob = noisy_operator['probabilities']
    
    for noisy_instruction in noisy_operator['instructions']:
        
        super_op = kraus_to_super(ideal_operation_kraus) # the exact ideal operator is always in the frount
        for i in range(len(noisy_instruction)):
            kraus_set = []
            if noisy_instruction[i]['name']=='kraus':
                for kraus_operator in noisy_instruction[i]['params']:
                    kraus_set.append(kraus_operator)
            elif noisy_instruction[i]['name']=='reset':
                kraus_set.append(np.array([[1,0],[0,0]],dtype=np.complex128))
                kraus_set.append(np.array([[0,1],[0,0]],dtype=np.complex128))
            else:
                kraus_operator = standard_gate_unitary(noisy_instruction[i]['name'])
                kraus_set.append(kraus_operator)
            super_op = super_op @ kraus_to_super(kraus_set)
            
        super_op_list.append(super_op)

    noisy_super_op = np.zeros(super_op.shape)
    for i in range(len(super_op_list)):
        noisy_super_op = noisy_super_op + super_op_list[i]*prob[i]
    noisy_super_op_list.append(noisy_super_op)
    


print(noisy_super_op_list)
    
id_ideal = QuantumCircuit(1)
id_ideal.id(0)

basis_noise_circ_id_id = QuantumCircuit(1)
basis_noise_circ_id_id.id(0)
basis_noise_circ_id_id.id(0)
basis_noise_circ_id_x = QuantumCircuit(1)
basis_noise_circ_id_x.id(0)
basis_noise_circ_id_x.x(0)
basis_noise_circ_id_y = QuantumCircuit(1)
basis_noise_circ_id_y.id(0)
basis_noise_circ_id_y.y(0)
basis_noise_circ_id_z = QuantumCircuit(1)
basis_noise_circ_id_z.id(0)
basis_noise_circ_id_z.z(0)
basis_noise_circs = [basis_noise_circ_id_id, basis_noise_circ_id_x, basis_noise_circ_id_y,basis_noise_circ_id_z]

sigma = 0.2
for i in range(int(1/sigma)-1):
    theta = (1+i)*sigma*2*np.pi
    basis_noise_circ_id_rz = QuantumCircuit(1)
    basis_noise_circ_id_rz.rz(theta,0)
    basis_noise_circs.append(basis_noise_circ_id_rz)
    noisy_super_op_list.append(kraus_to_super([RZGate(theta).to_matrix()]))
noisy_operations = [
    pec.NoisyOperation(circuit=c, channel_matrix=m)
    for c,m in zip(basis_noise_circs, noisy_super_op_list)
]

id_rep = find_optimal_representation(id_ideal, noisy_operations, tol=1e-8)
print(id_rep)



# test = QuantumCircuit(1)
# qc.