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
from qiskit.circuit.library import RZGate, SXGate
from qiskit.circuit.library.generalized_gates import PauliGate
from qiskit.extensions import UnitaryGate
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



noise_dict = pd.read_pickle('./NoiseModel/fakecairo.pkl')
print("noise_dict loaded")



single_qubit_noise_operator = []
two_qubits_noise_operator = []
two_graphs = [] # the map for cx

for qubit in range(27):
    single_qubit_noise_operator.append([])
    
for noise in noise_dict['errors']:
    if len(noise['gate_qubits'][0])==1:
        qubit = noise['gate_qubits'][0][0]
        single_qubit_noise_operator[qubit].append(noise)
    elif len(noise['gate_qubits'][0])==2:
        two_graphs.append(noise['gate_qubits'][0])
        two_qubits_noise_operator.append(noise)
        
# print(len(single_qubit_noise_operator))
# print(len(single_qubit_noise_operator[0]))
# print(len(two_qubits_noise_operator))

sigma = 0.2 # for rz

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
        theta = i*sigma*2*np.pi
        single_noisy_super_op_list[qubit].append(kraus_to_super([RZGate(theta).to_matrix()]))
        
    single_noisy_super_op_list[qubit].append(single_noisy_super_op_list[qubit][1] @ single_noisy_super_op_list[qubit][2]) # sx*x
    single_noisy_super_op_list[qubit].append(single_noisy_super_op_list[qubit][2] @ single_noisy_super_op_list[qubit][1]) # x*sx
    for i in range(int(1/sigma)):
        single_noisy_super_op_list[qubit].append(single_noisy_super_op_list[qubit][1] @ single_noisy_super_op_list[qubit][3+i]) # sx*rz
        single_noisy_super_op_list[qubit].append(single_noisy_super_op_list[qubit][3+i] @ single_noisy_super_op_list[qubit][1]) # rz*sx
        single_noisy_super_op_list[qubit].append(single_noisy_super_op_list[qubit][2] @ single_noisy_super_op_list[qubit][3+i]) # x*rz
        single_noisy_super_op_list[qubit].append(single_noisy_super_op_list[qubit][3+i] @ single_noisy_super_op_list[qubit][2]) # rz*x
single_sup_op_end = time.time()
print(f"super operators for single operations done. time consumption: {single_sup_op_end-single_sup_op_start}")

        

cx_sup_op_start = time.time()
        
# cx, cx*sx_0, cx*x_0, cx*id_1, cx*sx_1, cx*x_1, sx_0, x_0, rz's_0, (sx*x)_0, (sx*rz's)_0, (x*rz's)_1, sx_1, x_1, (rz's)_1, (sx*x)_1, (sx*rz's)_1, (x*rz's)_1, 

two_qubits_noisy_super_op_list = []
for index in range(len(two_graphs)):
    two_qubits_noisy_super_op_list.append([])
    qubit_0 = two_graphs[index][0]
    qubit_1 = two_graphs[index][1]
    
    for noisy_operator in two_qubits_noise_operator:
        ideal_operation_kraus = []
        for i in range(len(noisy_operator['operations'])):
            kraus_sets = []
            kraus_operator = standard_gate_unitary(noisy_operator['operations'][i])
            ideal_operation_kraus.append(kraus_operator)
    

    super_op_list = []
    prob = noisy_operator['probabilities']
    # Special attention!! in qiskit, due to its special order of qubits, the matrix of a union operator should be q1 @ q0
    for noisy_instruction in noisy_operator['instructions']:
        super_op = kraus_to_super(ideal_operation_kraus)
        
        for i in range(len(ideal_operation_kraus)):
            kraus_set = []
            if len(noisy_instruction[i]['qubits'])==2:
                if noisy_instruction[i]['name']=='pauli':
                    gate = PauliGate(noisy_instruction[i]['params'][0])
                    kraus_set.append(gate.to_matrix())
                elif noisy_instruction[i]['name']=='unitry':
                    gate = UnitaryGate(data=noisy_instruction[i]['params'][0])
                    kraus_set.append(gate.to_matrix())
                else:
                    print('warning!!!!!!!Two qubit error not consider!')

            else:
                if(noisy_instruction[i]['qubits'][0]==0):
                    if noisy_instruction[i]['name']=='' or noisy_instruction[i]['name']=='measure' or noisy_instruction[i]['name']=='reset':
                        continue
                    
                    elif noisy_instruction[i]['name']=='kraus':
                        for kraus_operator in noisy_instruction[i]['params']:
                            kraus_operator = np.kron(np.array([[1,0],[0,1]],dtype=complex),kraus_operator) 
                            kraus_set.append(kraus_operator)
                            
                    elif noisy_instruction[i]['name']=='reset':
                        kraus_set.append(np.kron(np.array([[1,0],[0,1]],dtype=complex),np.array([[1,0],[0,0]],dtype=np.complex128)))
                        kraus_set.append(np.kron(np.array([[1,0],[0,1]],dtype=complex),np.array([[0,1],[0,0]],dtype=np.complex128)))
                    else:
                        kraus_operator = np.kron(np.array([[1,0],[0,1]],dtype=complex),standard_gate_unitary(noisy_instruction[i]['name']))
                        kraus_set.append(kraus_operator)
                    
                else:
                    if noisy_instruction[i]['name']=='' or noisy_instruction[i]['name']=='measure' or noisy_instruction[i]['name']=='reset':
                        continue
                    elif noisy_instruction[i]['name']=='kraus':
                        for kraus_operator in noisy_instruction[i]['params']:
                            kraus_operator = np.kron(kraus_operator,np.array([[1,0],[0,1]],dtype=complex)) 
                            kraus_set.append(kraus_operator)
                            
                    elif noisy_instruction[i]['name']=='reset':
                        kraus_set.append(np.kron(np.array([[1,0],[0,0]],dtype=np.complex128),np.array([[1,0],[0,1]],dtype=complex)))
                        kraus_set.append(np.kron(np.array([[0,1],[0,0]],dtype=np.complex128),np.array([[1,0],[0,1]],dtype=complex)))
                    else:
                        kraus_operator = np.kron(standard_gate_unitary(noisy_instruction[i]['name']),np.array([[1,0],[0,1]],dtype=complex))
                        kraus_set.append(kraus_operator)
            super_op = super_op @ kraus_to_super(kraus_set)
        
        super_op_list.append(super_op)
            
    noisy_super_op = np.zeros(super_op.shape)
    for i in range(len(super_op_list)):
        noisy_super_op = noisy_super_op + prob[i]*super_op_list[i]
    two_qubits_noisy_super_op_list[index].append(noisy_super_op)
    
    two_qubits_noisy_super_op_list[index].append(two_qubits_noisy_super_op_list[index][0] @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),single_noisy_super_op_list[qubit_0][1])) #cx*sx_0
    two_qubits_noisy_super_op_list[index].append(two_qubits_noisy_super_op_list[index][0] @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),single_noisy_super_op_list[qubit_0][2])) #cx*x_0
    

    
    
    # #cx*ry_0
    # for i in range(int(1/sigma)):
    #     two_qubits_noisy_super_op_list[index].append(two_qubits_noisy_super_op_list[index][0] @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),kraus_to_super([RZGate(np.pi).to_matrix()])) @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),kraus_to_super([SXGate().to_matrix()])) @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),single_noisy_super_op_list[qubit_0][3+i]) @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),kraus_to_super([SXGate().to_matrix()])))
    
    # #cx*rx_0
    # for i in range(int(1/sigma)):
    #     two_qubits_noisy_super_op_list[index].append(two_qubits_noisy_super_op_list[index][0] @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),kraus_to_super([RZGate(np.pi/2).to_matrix()])) @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),kraus_to_super([SXGate().to_matrix()])) @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),single_noisy_super_op_list[qubit_0][3+i]) @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),kraus_to_super([SXGate().to_matrix()])) @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),kraus_to_super([RZGate(np.pi/2).to_matrix()])))
        
        
    
    for i in range(int(1/sigma)):
        two_qubits_noisy_super_op_list[index].append(two_qubits_noisy_super_op_list[index][0] @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),(single_noisy_super_op_list[qubit_0][1] @ single_noisy_super_op_list[qubit_0][3+i]))) #cx*(sx*rz)_0
        two_qubits_noisy_super_op_list[index].append(two_qubits_noisy_super_op_list[index][0] @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),(single_noisy_super_op_list[qubit_0][3+i] @ single_noisy_super_op_list[qubit_0][1]))) #cx*(rz*sx)_0
        two_qubits_noisy_super_op_list[index].append(two_qubits_noisy_super_op_list[index][0] @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),(single_noisy_super_op_list[qubit_0][2] @ single_noisy_super_op_list[qubit_0][3+i]))) #cx*(x*rz)_0
        two_qubits_noisy_super_op_list[index].append(two_qubits_noisy_super_op_list[index][0] @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),(single_noisy_super_op_list[qubit_0][3+i] @ single_noisy_super_op_list[qubit_0][2]))) #cx*(rz*x)_0
    
    
    two_qubits_noisy_super_op_list[index].append(two_qubits_noisy_super_op_list[index][0] @ np.kron(single_noisy_super_op_list[qubit_1][1],np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128))) #cx*sx_1
    two_qubits_noisy_super_op_list[index].append(two_qubits_noisy_super_op_list[index][0] @ np.kron(single_noisy_super_op_list[qubit_1][2], np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128))) #cx*x_1
    
    
    
    
    
    
    
    # #cx*ry_1
    # for i in range(int(1/sigma)):
    #     two_qubits_noisy_super_op_list[index].append(two_qubits_noisy_super_op_list[index][0] @ np.kron(kraus_to_super([RZGate(np.pi).to_matrix()]),np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(kraus_to_super([SXGate().to_matrix()]),np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(single_noisy_super_op_list[qubit_1][3+i],np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(kraus_to_super([SXGate().to_matrix()]),np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)))
        
    # #cx*rx_1
    # for i in range(int(1/sigma)):
    #     two_qubits_noisy_super_op_list[index].append(two_qubits_noisy_super_op_list[index][0] @ np.kron(kraus_to_super([RZGate(np.pi/2).to_matrix()]),np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(kraus_to_super([SXGate().to_matrix()]),np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(single_noisy_super_op_list[qubit_1][3+i],np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(kraus_to_super([SXGate().to_matrix()]),np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(kraus_to_super([RZGate(np.pi/2).to_matrix()]),np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)))
    
    
    
    for i in range(int(1/sigma)):
        two_qubits_noisy_super_op_list[index].append(two_qubits_noisy_super_op_list[index][0] @ np.kron((single_noisy_super_op_list[qubit_1][1] @ single_noisy_super_op_list[qubit_0][3+i]), np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128))) #cx*(sx*rz)_1
        two_qubits_noisy_super_op_list[index].append(two_qubits_noisy_super_op_list[index][0] @ np.kron((single_noisy_super_op_list[qubit_1][3+i] @ single_noisy_super_op_list[qubit_0][1]), np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128))) #cx*(rz*sx)_1
        two_qubits_noisy_super_op_list[index].append(two_qubits_noisy_super_op_list[index][0] @ np.kron((single_noisy_super_op_list[qubit_1][2] @ single_noisy_super_op_list[qubit_0][3+i]), np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128))) #cx*(x*rz)_1
        two_qubits_noisy_super_op_list[index].append(two_qubits_noisy_super_op_list[index][0] @ np.kron((single_noisy_super_op_list[qubit_1][3+i] @ single_noisy_super_op_list[qubit_0][2]), np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128))) #cx*(rz*x)_1
        
        
        
    for i in range(int(1/sigma)):
        for j in range(int(1/sigma)):
            # cx*(sx*rz)_0*(sx*rz)_1
            two_qubits_noisy_super_op_list[index].append(two_qubits_noisy_super_op_list[index][0] @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),single_noisy_super_op_list[qubit_0][1]) @ np.kron(single_noisy_super_op_list[qubit_1][1], np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(single_noisy_super_op_list[qubit_0][3+i], np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(single_noisy_super_op_list[qubit_1][3+j], np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)))
            # cx*(x*rz)_0*(x*rz)_1
            two_qubits_noisy_super_op_list[index].append(two_qubits_noisy_super_op_list[index][0] @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),single_noisy_super_op_list[qubit_0][2]) @ np.kron(single_noisy_super_op_list[qubit_1][2], np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(single_noisy_super_op_list[qubit_0][3+i], np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(single_noisy_super_op_list[qubit_1][3+j], np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)))
            # cx*(sx*rz)_0*(x*rz)_1
            two_qubits_noisy_super_op_list[index].append(two_qubits_noisy_super_op_list[index][0] @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),single_noisy_super_op_list[qubit_0][1]) @ np.kron(single_noisy_super_op_list[qubit_1][2], np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(single_noisy_super_op_list[qubit_0][3+i], np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(single_noisy_super_op_list[qubit_1][3+j], np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)))
            # cx*(s*rz)_0*(sx*rz)_1
            two_qubits_noisy_super_op_list[index].append(two_qubits_noisy_super_op_list[index][0] @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),single_noisy_super_op_list[qubit_0][2]) @ np.kron(single_noisy_super_op_list[qubit_1][1], np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(single_noisy_super_op_list[qubit_0][3+i], np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(single_noisy_super_op_list[qubit_1][3+j], np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)))
            
    
    
    

    for i in range(int(1/sigma)):
        for j in range(int(1/sigma)):
            for k in range(int(1/sigma)):
                rz = np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),single_noisy_super_op_list[qubit_0][3+i])
                ry = np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),kraus_to_super([RZGate(np.pi).to_matrix()])) @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),single_noisy_super_op_list[qubit_0][1]) @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),single_noisy_super_op_list[qubit_0][3+j]) @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),single_noisy_super_op_list[qubit_0][1])
                rx = np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),kraus_to_super([RZGate(np.pi/2).to_matrix()])) @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),single_noisy_super_op_list[qubit_0][1]) @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),single_noisy_super_op_list[qubit_0][3+k]) @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),single_noisy_super_op_list[qubit_0][1]) @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),kraus_to_super([RZGate(np.pi/2).to_matrix()]))
                two_qubits_noisy_super_op_list[index].append(two_qubits_noisy_super_op_list[index][0] @ rz @ ry @ rx) 
                
    #rz_1*ry_1*rx_1
    rotation_1 = []
    for i in range(int(1/sigma)):
        for j in range(int(1/sigma)):
            for k in range(int(1/sigma)):
                rz = np.kron(single_noisy_super_op_list[qubit_1][3+i], np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128))
                ry = np.kron(kraus_to_super([RZGate(np.pi).to_matrix()]),np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(single_noisy_super_op_list[qubit_1][1],np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(single_noisy_super_op_list[qubit_1][3+j],np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(single_noisy_super_op_list[qubit_1][1],np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128))
                rx = np.kron(kraus_to_super([RZGate(np.pi/2).to_matrix()]),np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(single_noisy_super_op_list[qubit_1][1],np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(single_noisy_super_op_list[qubit_1][3+k],np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(single_noisy_super_op_list[qubit_1][1],np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(kraus_to_super([RZGate(np.pi/2).to_matrix()]),np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128))
                
                two_qubits_noisy_super_op_list[index].append(two_qubits_noisy_super_op_list[index][0] @ rz @ ry @ rx)
    
 
    # single_qubit
    for i in range(len(single_noisy_super_op_list[qubit_0])):
        two_qubits_noisy_super_op_list[index].append(np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),single_noisy_super_op_list[qubit_0][i]))
        
    for i in range(len(single_noisy_super_op_list[qubit_1])):
        two_qubits_noisy_super_op_list[index].append(np.kron(single_noisy_super_op_list[qubit_1][i],np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)))

cx_sup_op_end = time.time()
print(f"super operators for cx operations done. time consumption: {cx_sup_op_end-cx_sup_op_start}")
    
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
        theta = j*sigma*2*np.pi
        basis_noise_circ_rz = QuantumCircuit(27)
        basis_noise_circ_rz.rz(theta,i)
        basis_noise_circs.append(basis_noise_circ_rz)
    basis_noise_circs.append(basis_noise_circ_sx_x)
    basis_noise_circs.append(basis_noise_circ_x_sx)
    
    # with_rz
    for j in range(int(1/sigma)):
        theta = j*sigma*2*np.pi
        basis_noise_circ_sx_rz = QuantumCircuit(27)
        basis_noise_circ_sx_rz.sx(i)
        basis_noise_circ_sx_rz.rz(theta,i)
        basis_noise_circs.append(basis_noise_circ_sx_rz)
        basis_noise_circ_rz_sx = QuantumCircuit(27)
        basis_noise_circ_rz_sx.rz(theta,i)
        basis_noise_circ_rz_sx.sx(i)
        basis_noise_circs.append(basis_noise_circ_rz_sx)
        basis_noise_circ_x_rz = QuantumCircuit(27)
        basis_noise_circ_x_rz.x(i)
        basis_noise_circ_x_rz.rz(theta,i)
        basis_noise_circs.append(basis_noise_circ_x_rz)
        basis_noise_circ_rz_x = QuantumCircuit(27)
        basis_noise_circ_rz_x.rz(theta,i)
        basis_noise_circ_rz_x.x(i)
        basis_noise_circs.append(basis_noise_circ_rz_x)
    
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

    
    id_rep = find_optimal_representation(ideal_id, noisy_operations, tol=1e-8)
    # print(f'id_rep for qubit {i} is\n\n{id_rep}')
    
    sx_rep = find_optimal_representation(ideal_sx, noisy_operations, tol=1e-8)
    # print(f'sx_rep for qubit {i} is\n\n{sx_rep}')
    
    x_rep = find_optimal_representation(ideal_x, noisy_operations, tol=1e-8, is_qubit_dependent=True)
    # print(f'x_rep for qubit {i} is\n\n{x_rep}')
    # print('\n\n')
    single_rep.append([id_rep, sx_rep, x_rep])
single_rep_end = time.time()
print(f"representations for single qubit operation found. time consumption: {single_rep_end-single_rep_start}")
    
        
cx_rep_start = time.time()
cx_rep = []
for i in range(len(two_graphs)):
    cx_rep_i_start = time.time()
    basis_noise_circs = []
    
    basis_noise_circ_cx = QuantumCircuit(27)
    basis_noise_circ_cx.cx(two_graphs[i][0],two_graphs[i][1])
    
    # 0
    basis_noise_circ_cx_1none_0sx = QuantumCircuit(27)
    basis_noise_circ_cx_1none_0sx.cx(two_graphs[i][0], two_graphs[i][1])
    basis_noise_circ_cx_1none_0sx.sx(two_graphs[i][0])
    basis_noise_circ_cx_1none_0sx.rz(0, two_graphs[i][1])
    
    basis_noise_circ_cx_1none_0x = QuantumCircuit(27)
    basis_noise_circ_cx_1none_0x.cx(two_graphs[i][0], two_graphs[i][1])
    basis_noise_circ_cx_1none_0x.x(two_graphs[i][0])
    basis_noise_circ_cx_1none_0x.rz(0, two_graphs[i][1])
    

    
    basis_noise_circs.append(basis_noise_circ_cx)
    basis_noise_circs.append(basis_noise_circ_cx_1none_0sx) 
    basis_noise_circs.append(basis_noise_circ_cx_1none_0x)
    
    
    
    
        
        
    # cx_withrz_0
    for j in range(int(1/sigma)):
        theta = j*sigma*2*np.pi
        basis_noise_circ_cx_1none_0sxrz = QuantumCircuit(27)
        basis_noise_circ_cx_1none_0sxrz.cx(two_graphs[i][0], two_graphs[i][1])
        basis_noise_circ_cx_1none_0sxrz.sx(two_graphs[i][0])
        basis_noise_circ_cx_1none_0sxrz.rz(theta, two_graphs[i][0])
        basis_noise_circ_cx_1none_0sxrz.rz(0,two_graphs[i][1])
        basis_noise_circs.append(basis_noise_circ_cx_1none_0sxrz)
        basis_noise_circ_cx_1none_0rzsx = QuantumCircuit(27)
        basis_noise_circ_cx_1none_0rzsx.cx(two_graphs[i][0], two_graphs[i][1])
        basis_noise_circ_cx_1none_0rzsx.rz(theta, two_graphs[i][0])
        basis_noise_circ_cx_1none_0rzsx.sx(two_graphs[i][0])
        basis_noise_circ_cx_1none_0rzsx.rz(0,two_graphs[i][1])
        basis_noise_circs.append(basis_noise_circ_cx_1none_0rzsx)
        basis_noise_circ_cx_1none_0xrz = QuantumCircuit(27)
        basis_noise_circ_cx_1none_0xrz.cx(two_graphs[i][0], two_graphs[i][1])
        basis_noise_circ_cx_1none_0xrz.x(two_graphs[i][0])
        basis_noise_circ_cx_1none_0xrz.rz(theta, two_graphs[i][0])
        basis_noise_circ_cx_1none_0xrz.rz(0,two_graphs[i][1])
        basis_noise_circs.append(basis_noise_circ_cx_1none_0xrz)
        basis_noise_circ_cx_1none_0rzx = QuantumCircuit(27)
        basis_noise_circ_cx_1none_0rzx.cx(two_graphs[i][0], two_graphs[i][1])
        basis_noise_circ_cx_1none_0rzx.rz(theta, two_graphs[i][0])
        basis_noise_circ_cx_1none_0rzx.x(two_graphs[i][0])
        basis_noise_circ_cx_1none_0rzx.rz(0,two_graphs[i][1])
        basis_noise_circs.append(basis_noise_circ_cx_1none_0rzx)
          
          
          
          
          
    # 1
    basis_noise_circ_cx_1sx_0none = QuantumCircuit(27)
    basis_noise_circ_cx_1sx_0none.cx(two_graphs[i][0], two_graphs[i][1])
    basis_noise_circ_cx_1sx_0none.sx(two_graphs[i][1])
    basis_noise_circ_cx_1sx_0none.rz(0, two_graphs[i][0])
    basis_noise_circs.append(basis_noise_circ_cx_1sx_0none)
    
    basis_noise_circ_cx_1x_0none = QuantumCircuit(27)
    basis_noise_circ_cx_1x_0none.cx(two_graphs[i][0], two_graphs[i][1])
    basis_noise_circ_cx_1x_0none.x(two_graphs[i][1])
    basis_noise_circ_cx_1x_0none.rz(0, two_graphs[i][0])
    basis_noise_circs.append(basis_noise_circ_cx_1x_0none)
    
    


    # cx_withrz_1
    for j in range(int(1/sigma)):
            theta = j*sigma*2*np.pi
            basis_noise_circ_cx_1sxrz_0none = QuantumCircuit(27)
            basis_noise_circ_cx_1sxrz_0none.cx(two_graphs[i][0], two_graphs[i][1])
            basis_noise_circ_cx_1sxrz_0none.sx(two_graphs[i][1])
            basis_noise_circ_cx_1sxrz_0none.rz(theta, two_graphs[i][1])
            basis_noise_circ_cx_1sxrz_0none.rz(0,two_graphs[i][0])
            basis_noise_circs.append(basis_noise_circ_cx_1sxrz_0none)
            basis_noise_circ_cx_1rzsx_0none = QuantumCircuit(27)
            basis_noise_circ_cx_1rzsx_0none.cx(two_graphs[i][0], two_graphs[i][1])
            basis_noise_circ_cx_1rzsx_0none.rz(theta, two_graphs[i][1])
            basis_noise_circ_cx_1rzsx_0none.sx(two_graphs[i][1])
            basis_noise_circ_cx_1rzsx_0none.rz(0,two_graphs[i][0])
            basis_noise_circs.append(basis_noise_circ_cx_1rzsx_0none)
            basis_noise_circ_cx_1xrz_0none = QuantumCircuit(27)
            basis_noise_circ_cx_1xrz_0none.cx(two_graphs[i][0], two_graphs[i][1])
            basis_noise_circ_cx_1xrz_0none.x(two_graphs[i][1])
            basis_noise_circ_cx_1xrz_0none.rz(theta, two_graphs[i][1])
            basis_noise_circ_cx_1xrz_0none.rz(0,two_graphs[i][0])
            basis_noise_circs.append(basis_noise_circ_cx_1xrz_0none)
            basis_noise_circ_cx_1rzx_0none = QuantumCircuit(27)
            basis_noise_circ_cx_1rzx_0none.cx(two_graphs[i][0], two_graphs[i][1])
            basis_noise_circ_cx_1rzx_0none.rz(theta, two_graphs[i][1])
            basis_noise_circ_cx_1rzx_0none.x(two_graphs[i][1])
            basis_noise_circ_cx_1rzx_0none.rz(0,two_graphs[i][0])
            basis_noise_circs.append(basis_noise_circ_cx_1rzx_0none)    
    
    # cx_withrz_0and1
    for j in range(int(1/sigma)):
        for k in range(int(1/sigma)):
            theta_0 = j*sigma*2*np.pi
            theta_1 = k*sigma*2*np.pi

            basis_noise_circ_cx_1sxrz_0sxrz = QuantumCircuit(27)
            basis_noise_circ_cx_1sxrz_0sxrz.cx(two_graphs[i][0], two_graphs[i][1])
            basis_noise_circ_cx_1sxrz_0sxrz.sx(two_graphs[i][0])
            basis_noise_circ_cx_1sxrz_0sxrz.sx(two_graphs[i][1])
            basis_noise_circ_cx_1sxrz_0sxrz.rz(theta_0, two_graphs[i][0])
            basis_noise_circ_cx_1sxrz_0sxrz.rz(theta_1, two_graphs[i][1])
            basis_noise_circs.append(basis_noise_circ_cx_1sxrz_0sxrz)
    
            basis_noise_circ_cx_1xrz_0xrz = QuantumCircuit(27)
            basis_noise_circ_cx_1xrz_0xrz.cx(two_graphs[i][0], two_graphs[i][1])
            basis_noise_circ_cx_1xrz_0xrz.x(two_graphs[i][0])
            basis_noise_circ_cx_1xrz_0xrz.x(two_graphs[i][1])
            basis_noise_circ_cx_1xrz_0xrz.rz(theta_0, two_graphs[i][0])
            basis_noise_circ_cx_1xrz_0xrz.rz(theta_1, two_graphs[i][1])
            basis_noise_circs.append(basis_noise_circ_cx_1xrz_0xrz)   
    
            basis_noise_circ_cx_1xrz_0sxrz = QuantumCircuit(27)
            basis_noise_circ_cx_1xrz_0sxrz.cx(two_graphs[i][0], two_graphs[i][1])
            basis_noise_circ_cx_1xrz_0sxrz.sx(two_graphs[i][0])
            basis_noise_circ_cx_1xrz_0sxrz.x(two_graphs[i][1])
            basis_noise_circ_cx_1xrz_0sxrz.rz(theta_0, two_graphs[i][0])
            basis_noise_circ_cx_1xrz_0sxrz.rz(theta_1, two_graphs[i][1])
            basis_noise_circs.append(basis_noise_circ_cx_1xrz_0sxrz)   
    
            basis_noise_circ_cx_1sxrz_0xrz = QuantumCircuit(27)
            basis_noise_circ_cx_1sxrz_0xrz.cx(two_graphs[i][0], two_graphs[i][1])
            basis_noise_circ_cx_1sxrz_0xrz.x(two_graphs[i][0])
            basis_noise_circ_cx_1sxrz_0xrz.sx(two_graphs[i][1])
            basis_noise_circ_cx_1sxrz_0xrz.rz(theta_0, two_graphs[i][0])
            basis_noise_circ_cx_1sxrz_0xrz.rz(theta_1, two_graphs[i][1])
            basis_noise_circs.append(basis_noise_circ_cx_1sxrz_0xrz)   

  
  
    # cx*rot1*rot0
    for j in range(int(1/sigma)):
        for k in range(int(1/sigma)):
            for l in range(int(1/sigma)):
                theta_z_0 = j*sigma*2*np.pi
                theta_y_0 = k*sigma*2*np.pi
                theta_x_0 = l*sigma*2*np.pi
                basis_noise_circ_cx_1none_0rot = QuantumCircuit(27)
                basis_noise_circ_cx_1none_0rot.cx(two_graphs[i][0], two_graphs[i][1])
                basis_noise_circ_cx_1none_0rot.rz(theta_z_0, two_graphs[i][0])
                basis_noise_circ_cx_1none_0rot.rz(np.pi, two_graphs[i][0])
                basis_noise_circ_cx_1none_0rot.sx(two_graphs[i][0])
                basis_noise_circ_cx_1none_0rot.rz(theta_y_0, two_graphs[i][0])
                basis_noise_circ_cx_1none_0rot.sx(two_graphs[i][0])
                basis_noise_circ_cx_1none_0rot.rz(np.pi/2, two_graphs[i][0])
                basis_noise_circ_cx_1none_0rot.sx(two_graphs[i][0])
                basis_noise_circ_cx_1none_0rot.rz(theta_x_0, two_graphs[i][0])
                basis_noise_circ_cx_1none_0rot.sx(two_graphs[i][0])
                basis_noise_circ_cx_1none_0rot.rz(np.pi/2, two_graphs[i][0])
                basis_noise_circ_cx_1none_0rot.rz(0, two_graphs[i][0])
                basis_noise_circs.append(basis_noise_circ_cx_1none_0rot)


    for j in range(int(1/sigma)):
        for k in range(int(1/sigma)):
            for l in range(int(1/sigma)):
                theta_z_1 = j*sigma*2*np.pi
                theta_y_1 = k*sigma*2*np.pi
                theta_x_1 = l*sigma*2*np.pi
                basis_noise_circ_cx_1rot_0none = QuantumCircuit(27)
                basis_noise_circ_cx_1rot_0none.rz(theta_z_1, two_graphs[i][1])
                basis_noise_circ_cx_1rot_0none.rz(np.pi, two_graphs[i][1])
                basis_noise_circ_cx_1rot_0none.sx(two_graphs[i][1])
                basis_noise_circ_cx_1rot_0none.rz(theta_y_1, two_graphs[i][1])
                basis_noise_circ_cx_1rot_0none.sx(two_graphs[i][1])
                basis_noise_circ_cx_1rot_0none.rz(np.pi/2, two_graphs[i][1])
                basis_noise_circ_cx_1rot_0none.sx(two_graphs[i][1])
                basis_noise_circ_cx_1rot_0none.rz(theta_x_1, two_graphs[i][1])
                basis_noise_circ_cx_1rot_0none.sx(two_graphs[i][1])
                basis_noise_circ_cx_1rot_0none.rz(np.pi/2, two_graphs[i][1])
                basis_noise_circ_cx_1rot_0none.rz(0, two_graphs[i][0])
                basis_noise_circs.append(basis_noise_circ_cx_1rot_0none)
                
                               
                
    
    
    
    
    
    # single*id 0
    basis_noise_circ_1none_0id = QuantumCircuit(27)
    basis_noise_circ_1none_0id.id(two_graphs[i][0])
    basis_noise_circ_1none_0id.rz(0,two_graphs[i][1])
    basis_noise_circ_1none_0sx = QuantumCircuit(27)
    basis_noise_circ_1none_0sx.sx(two_graphs[i][0])
    basis_noise_circ_1none_0sx.rz(0,two_graphs[i][1])
    basis_noise_circ_1none_0x = QuantumCircuit(27)
    basis_noise_circ_1none_0x.x(two_graphs[i][0])
    basis_noise_circ_1none_0x.rz(0,two_graphs[i][1])
    basis_noise_circs.append(basis_noise_circ_1none_0id)    
    basis_noise_circs.append(basis_noise_circ_1none_0sx)    
    basis_noise_circs.append(basis_noise_circ_1none_0x)    
    for j in range(int(1/sigma)):
        theta = j*sigma*2*np.pi
        basis_noise_circ_1none_0rz = QuantumCircuit(27)
        basis_noise_circ_1none_0rz.rz(theta,two_graphs[i][0])
        basis_noise_circ_1none_0rz.rz(0,two_graphs[i][1])
        basis_noise_circs.append(basis_noise_circ_1none_0rz)
        
    basis_noise_circ_1none_0sxx = QuantumCircuit(27)
    basis_noise_circ_1none_0sxx.sx(two_graphs[i][0])
    basis_noise_circ_1none_0sxx.x(two_graphs[i][0])
    basis_noise_circ_1none_0sxx.rz(0,two_graphs[i][1])
    basis_noise_circs.append(basis_noise_circ_1none_0sxx)    
        
    basis_noise_circ_1none_0xsx = QuantumCircuit(27)
    basis_noise_circ_1none_0xsx.x(two_graphs[i][0])
    basis_noise_circ_1none_0xsx.sx(two_graphs[i][0])
    basis_noise_circ_1none_0xsx.rz(0,two_graphs[i][1])
    basis_noise_circs.append(basis_noise_circ_1none_0xsx)    
    
    # with_rz_0
    for j in range(int(1/sigma)):
        theta = j*sigma*2*np.pi
        basis_noise_circ_1none_0sxrz = QuantumCircuit(27)
        basis_noise_circ_1none_0sxrz.sx(two_graphs[i][0])
        basis_noise_circ_1none_0sxrz.rz(theta,two_graphs[i][0])
        basis_noise_circ_1none_0sxrz.rz(0,two_graphs[i][1])
        basis_noise_circs.append(basis_noise_circ_1none_0sxrz)
        basis_noise_circ_1none_0rzsx = QuantumCircuit(27)
        basis_noise_circ_1none_0rzsx.rz(theta,two_graphs[i][0])
        basis_noise_circ_1none_0rzsx.sx(two_graphs[i][0])
        basis_noise_circ_1none_0rzsx.rz(0,two_graphs[i][1])
        basis_noise_circs.append(basis_noise_circ_1none_0rzsx)
        basis_noise_circ_1none_0xrz = QuantumCircuit(27)
        basis_noise_circ_1none_0xrz.x(two_graphs[i][0])
        basis_noise_circ_1none_0xrz.rz(theta,two_graphs[i][0])
        basis_noise_circ_1none_0xrz.rz(0,two_graphs[i][1])
        basis_noise_circs.append(basis_noise_circ_1none_0xrz)
        basis_noise_circ_1none_0rzx = QuantumCircuit(27)
        basis_noise_circ_1none_0rzx.rz(theta,two_graphs[i][0])
        basis_noise_circ_1none_0rzx.x(two_graphs[i][0])
        basis_noise_circ_1none_0rzx.rz(0,two_graphs[i][1])
        basis_noise_circs.append(basis_noise_circ_1none_0rzx)

    
    
    
    # single*id 1
    basis_noise_circ_1id_0none = QuantumCircuit(27)
    basis_noise_circ_1id_0none.id(two_graphs[i][1])
    basis_noise_circ_1id_0none.rz(0,two_graphs[i][0])
    basis_noise_circ_1sx_0none = QuantumCircuit(27)
    basis_noise_circ_1sx_0none.sx(two_graphs[i][1])
    basis_noise_circ_1sx_0none.rz(0,two_graphs[i][0])
    basis_noise_circ_1x_0none = QuantumCircuit(27)
    basis_noise_circ_1x_0none.x(two_graphs[i][1])
    basis_noise_circ_1x_0none.rz(0,two_graphs[i][0])
    basis_noise_circs.append(basis_noise_circ_1id_0none)    
    basis_noise_circs.append(basis_noise_circ_1sx_0none)    
    basis_noise_circs.append(basis_noise_circ_1x_0none)
    
    for j in range(int(1/sigma)):
        theta = j*sigma*2*np.pi
        basis_noise_circ_1rz_0none = QuantumCircuit(27)
        basis_noise_circ_1rz_0none.rz(theta,two_graphs[i][1])
        basis_noise_circ_1rz_0none.rz(0,two_graphs[i][0])
        basis_noise_circs.append(basis_noise_circ_1rz_0none)
        
    basis_noise_circ_1sxx_0none = QuantumCircuit(27)
    basis_noise_circ_1sxx_0none.sx(two_graphs[i][1])
    basis_noise_circ_1sxx_0none.x(two_graphs[i][1])
    basis_noise_circ_1sxx_0none.rz(0,two_graphs[i][0])
    basis_noise_circs.append(basis_noise_circ_1sxx_0none)    
        
    basis_noise_circ_1xsx_0none = QuantumCircuit(27)
    basis_noise_circ_1xsx_0none.x(two_graphs[i][1])
    basis_noise_circ_1xsx_0none.sx(two_graphs[i][1])
    basis_noise_circ_1xsx_0none.rz(0,two_graphs[i][0])
    basis_noise_circs.append(basis_noise_circ_1xsx_0none)    
    
    # with_rz_1
    for j in range(int(1/sigma)):
        theta = j*sigma*2*np.pi
        basis_noise_circ_1sxrz_0none = QuantumCircuit(27)
        basis_noise_circ_1sxrz_0none.sx(two_graphs[i][1])
        basis_noise_circ_1sxrz_0none.rz(theta,two_graphs[i][1])
        basis_noise_circ_1sxrz_0none.rz(0,two_graphs[i][0])
        basis_noise_circs.append(basis_noise_circ_1sxrz_0none)
        basis_noise_circ_1rzsx_0none = QuantumCircuit(27)
        basis_noise_circ_1rzsx_0none.rz(theta,two_graphs[i][1])
        basis_noise_circ_1rzsx_0none.sx(two_graphs[i][1])
        basis_noise_circ_1rzsx_0none.rz(0,two_graphs[i][0])
        basis_noise_circs.append(basis_noise_circ_1rzsx_0none)
        basis_noise_circ_1xrz_0none = QuantumCircuit(27)
        basis_noise_circ_1xrz_0none.x(two_graphs[i][1])
        basis_noise_circ_1xrz_0none.rz(theta,two_graphs[i][1])
        basis_noise_circ_1xrz_0none.rz(0,two_graphs[i][0])
        basis_noise_circs.append(basis_noise_circ_1xrz_0none)
        basis_noise_circ_1rzx_0none = QuantumCircuit(27)
        basis_noise_circ_1rzx_0none.rz(theta,two_graphs[i][1])
        basis_noise_circ_1rzx_0none.x(two_graphs[i][1])
        basis_noise_circ_1rzx_0none.rz(0,two_graphs[i][0])
        basis_noise_circs.append(basis_noise_circ_1rzx_0none)


    
    noisy_operations = [
        pec.NoisyOperation(circuit=c, channel_matrix=m)
        for c,m in zip(basis_noise_circs, two_qubits_noisy_super_op_list[i])
    ]
    ideal_cx = QuantumCircuit(27)
    ideal_cx.cx(two_graphs[i][0],two_graphs[i][1])
    
    cx_rep_i_mid = time.time()
    print(f"circuits constructed for the {i}'s pair. time consumption: {cx_rep_i_mid-cx_rep_i_start}")
    rep = find_optimal_representation(ideal_cx, noisy_operations, tol=1e-4, is_qubit_dependent=True)
    print(f'cx_rep for num {i} qubits {two_graphs[0]},{two_graphs[1]} is\n\n{rep}')
    print('\n\n')
    cx_rep.append(rep)
    cx_rep_i_end = time.time()
    print(f"representation for the {i}'s pair found. time consumption: {cx_rep_i_end-cx_rep_i_mid}")

cx_rep_end = time.time()
print(f"representations for all cx operation found. time consumption: {cx_rep_end-cx_rep_start}")

print(f"Done! total time: {cx_rep_end-single_sup_op_start}")

   
    
pdb.set_trace()
    
#TODO: read out error
    
