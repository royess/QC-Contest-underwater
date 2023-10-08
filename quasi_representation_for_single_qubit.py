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
from cirq import kraus
import pickle

from qiskit.circuit.library import RZGate, SXGate
from qiskit.circuit.library.generalized_gates import PauliGate


import time
import copy
import pdb



def standard_gate_unitary(name):
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

start = time.time()
noise_model_name = ['fakecairo', 'fakekolkata', 'fakemontreal']
representations = {}
for noise_name in noise_model_name:
    noise_dict = pd.read_pickle('./NoiseModel/'+noise_name+'.pkl')
    single_qubit_noise_operator = []

    for qubit in range(27):
        single_qubit_noise_operator.append([])
        
    for noise in noise_dict['errors']:
        if len(noise['gate_qubits'][0])==1:
            qubit = noise['gate_qubits'][0][0]
            single_qubit_noise_operator[qubit].append(noise)

    sigma = 0.125 # for rz

    single_sup_op_start = time.time()
    # id, sx, x, rz's, sx*x, sx*rz's, x*rz's, 
    single_noisy_super_op_list = []
    for qubit in range(27):
        single_noisy_super_op_list.append([])

        for noisy_operator in single_qubit_noise_operator[qubit]:
            if noisy_operator['operations'][0]=='' or noisy_operator['operations'][0]=='measure' or noisy_operator['operations'][0]=='reset':
                continue
            
            ideal_operation_kraus = []
            for i in range(len(noisy_operator['operations'])):
                # print(noisy_operator['operations'][i])
                kraus_sets = []
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
            single_noisy_super_op_list[qubit].append(noisy_super_op)
            
        for i in range(int(1/sigma)):
            theta = i*sigma*np.pi
            single_noisy_super_op_list[qubit].append(kraus_to_super([RZGate(theta).to_matrix()])) # rz
            single_noisy_super_op_list[qubit].append(kraus_to_super([RZGate(-np.pi).to_matrix()]) @ single_noisy_super_op_list[qubit][1] @ kraus_to_super([RZGate(np.pi-theta).to_matrix()]) @ single_noisy_super_op_list[qubit][1]) # ry
            single_noisy_super_op_list[qubit].append(kraus_to_super([RZGate(np.pi/2).to_matrix()]) @ single_noisy_super_op_list[qubit][1] @ kraus_to_super([RZGate(np.pi+theta).to_matrix()]) @ single_noisy_super_op_list[qubit][1] @ kraus_to_super([RZGate(np.pi/2).to_matrix()])) # ry
            
            
        single_noisy_super_op_list[qubit].append(single_noisy_super_op_list[qubit][1] @ single_noisy_super_op_list[qubit][2]) # sx*x
        single_noisy_super_op_list[qubit].append(single_noisy_super_op_list[qubit][2] @ single_noisy_super_op_list[qubit][1]) # x*sx
        
        for i in range(int(1/sigma)):
            single_noisy_super_op_list[qubit].append(single_noisy_super_op_list[qubit][1] @ single_noisy_super_op_list[qubit][3+i*3]) # sx*rz
            single_noisy_super_op_list[qubit].append(single_noisy_super_op_list[qubit][1] @ single_noisy_super_op_list[qubit][3+i*3+1]) # sx*ry
            single_noisy_super_op_list[qubit].append(single_noisy_super_op_list[qubit][1] @ single_noisy_super_op_list[qubit][3+i*3+2]) # sx*rx
            single_noisy_super_op_list[qubit].append(single_noisy_super_op_list[qubit][2] @ single_noisy_super_op_list[qubit][3+i*3]) # x*rz
            single_noisy_super_op_list[qubit].append(single_noisy_super_op_list[qubit][2] @ single_noisy_super_op_list[qubit][3+i*3+1]) # x*ry
            single_noisy_super_op_list[qubit].append(single_noisy_super_op_list[qubit][2] @ single_noisy_super_op_list[qubit][3+i*3+2]) # x*rx
    
    single_sup_op_end = time.time()
    print(f"super operators for single operations done. time consumption: {single_sup_op_end-single_sup_op_start}")
       
    single_rep_start = time.time()
    single_rep = []
    for i in range(27):

        basis_noise_circ_id = QuantumCircuit(27)
        basis_noise_circ_id.id(i)
        basis_noise_circ_sx = QuantumCircuit(27)
        basis_noise_circ_sx.sx(i)
        basis_noise_circ_x = QuantumCircuit(27)
        basis_noise_circ_x.x(i)
        basis_noise_circ_sx_x = QuantumCircuit(27)
        basis_noise_circ_sx_x.sx(i)
        basis_noise_circ_sx_x.x(i)
        basis_noise_circ_x_sx = QuantumCircuit(27)
        basis_noise_circ_x_sx.x(i)
        basis_noise_circ_x_sx.sx(i)

        basis_noise_circs = [basis_noise_circ_id, basis_noise_circ_sx, basis_noise_circ_x]
        for j in range(int(1/sigma)):
            theta = j*sigma*np.pi
            basis_noise_circ_rz = QuantumCircuit(27)
            basis_noise_circ_rz.rz(theta,i)
            basis_noise_circs.append(basis_noise_circ_rz)
            
            basis_noise_circ_ry = QuantumCircuit(27)
            basis_noise_circ_ry.rz(-np.pi,i)
            basis_noise_circ_ry.sx(i)
            basis_noise_circ_ry.rz(np.pi-theta,i)
            basis_noise_circ_ry.sx(i)
            basis_noise_circs.append(basis_noise_circ_ry)
            
            basis_noise_circ_rx = QuantumCircuit(27)
            basis_noise_circ_rx.rz(np.pi/2,i)
            basis_noise_circ_rx.sx(i)
            basis_noise_circ_rx.rz(np.pi+theta,i)
            basis_noise_circ_rx.sx(i)
            basis_noise_circ_rx.rz(np.pi/2,i)
            basis_noise_circs.append(basis_noise_circ_rx)
            
        basis_noise_circs.append(basis_noise_circ_sx_x)
        basis_noise_circs.append(basis_noise_circ_x_sx)
        
        # with_rz
        for j in range(int(1/sigma)):
            theta = j*sigma*np.pi
            basis_noise_circ_sx_rz = QuantumCircuit(27)
            basis_noise_circ_sx_rz.sx(i)
            basis_noise_circ_sx_rz.rz(theta,i)
            basis_noise_circs.append(basis_noise_circ_sx_rz)
            
            basis_noise_circ_sx_ry = QuantumCircuit(27)
            basis_noise_circ_sx_ry.sx(i)
            basis_noise_circ_sx_ry.rz(-np.pi,i)
            basis_noise_circ_sx_ry.sx(i)
            basis_noise_circ_sx_ry.rz(np.pi-theta,i)
            basis_noise_circ_sx_ry.rz(theta,i)
            basis_noise_circs.append(basis_noise_circ_sx_ry)
            
            basis_noise_circ_sx_rx = QuantumCircuit(27)
            basis_noise_circ_sx_rx.sx(i)
            basis_noise_circ_sx_rx.rz(np.pi/2,i)
            basis_noise_circ_sx_rx.sx(i)
            basis_noise_circ_sx_rx.rz(np.pi+theta,i)
            basis_noise_circ_sx_rx.sx(i)
            basis_noise_circ_sx_rx.rz(np.pi/2,i)
            basis_noise_circs.append(basis_noise_circ_sx_rx)
            
            basis_noise_circ_x_rz = QuantumCircuit(27)
            basis_noise_circ_x_rz.x(i)
            basis_noise_circ_x_rz.rz(theta,i)
            basis_noise_circs.append(basis_noise_circ_x_rz)
            
            basis_noise_circ_x_ry = QuantumCircuit(27)
            basis_noise_circ_x_ry.sx(i)
            basis_noise_circ_x_ry.rz(-np.pi,i)
            basis_noise_circ_x_ry.sx(i)
            basis_noise_circ_x_ry.rz(np.pi-theta,i)
            basis_noise_circ_x_ry.rz(theta,i)
            basis_noise_circs.append(basis_noise_circ_x_ry)
            
            basis_noise_circ_x_rx = QuantumCircuit(27)
            basis_noise_circ_x_rx.sx(i)
            basis_noise_circ_x_rx.rz(np.pi/2,i)
            basis_noise_circ_x_rx.sx(i)
            basis_noise_circ_x_rx.rz(np.pi+theta,i)
            basis_noise_circ_x_rx.sx(i)
            basis_noise_circ_x_rx.rz(np.pi/2,i)
            basis_noise_circs.append(basis_noise_circ_x_rx)

        
        noisy_operations = [
            pec.NoisyOperation(circuit=c, channel_matrix=m)
            for c,m in zip(basis_noise_circs, single_noisy_super_op_list[i])
        ]
        ideal_id = QuantumCircuit(27)
        ideal_id.id(i)
        ideal_sx = QuantumCircuit(27)
        ideal_sx.sx(i)
        ideal_x = QuantumCircuit(27)
        ideal_x.x(i)

        initial_guess_id = np.zeros(len(single_noisy_super_op_list[i]))
        initial_guess_id[0] = 1
        initial_guess_sx = np.zeros(len(single_noisy_super_op_list[i]))
        initial_guess_sx[0] = 0.6
        initial_guess_sx[1] = 0.4
        initial_guess_x = np.zeros(len(single_noisy_super_op_list[i]))
        initial_guess_x[0] = 0.7
        initial_guess_x[2] = 0.3
        # pdb.set_trace()
        id_rep = find_optimal_representation(ideal_id, noisy_operations, tol=1e-8, is_qubit_dependent=True, initial_guess = initial_guess_id)
        # print(f'id_rep for qubit {i} is\n\n{id_rep}')
        
        sx_rep = find_optimal_representation(ideal_sx, noisy_operations, tol=1e-8, is_qubit_dependent=True, initial_guess = initial_guess_sx)
        # print(f'sx_rep for qubit {i} is\n\n{sx_rep}')
        
        x_rep = find_optimal_representation(ideal_x, noisy_operations, tol=1e-8, is_qubit_dependent=True, initial_guess = initial_guess_x)
        # print(f'x_rep for qubit {i} is\n\n{x_rep}')
        # print('\n\n')
        single_rep.append(id_rep)
        single_rep.append(sx_rep)
        single_rep.append(x_rep)
        # pdb.set_trace()
    single_rep_end = time.time()
    print(f"representations for single qubit operation found. time consumption: {single_rep_end-single_rep_start}")



    print(f"Done! total time for noise_model_{noise_name}: {single_rep_end-single_sup_op_start}")

    representations[noise_name] = single_rep


file_path = './representations.pkl'
print(f'saving representations dict to {file_path}')
with open(file_path, "wb") as f: 
    pickle.dump(representations, f, protocol = pickle.HIGHEST_PROTOCOL)     
    f.close()

end = time.time()
print(f'All Done! time: {end-start}')

# pdb.set_trace()

pec_value, pec_data = pec.execute_with_pec(
    circuit,
    executor,
    observable=None, # In this example the observable is implicit in the executor
    representations=representations,
    num_samples=5, # Number of PEC samples
    random_state=0, # Seed for reproducibility of PEC sampling
    full_output=True, # Return "pec_data" in addition to "pec_value"
)