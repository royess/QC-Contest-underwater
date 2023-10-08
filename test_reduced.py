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

from qiskit.circuit.library import RZGate, SXGate
from qiskit.circuit.library.generalized_gates import PauliGate
from qiskit.extensions import UnitaryGate
from typing import List, Optional, cast
import numpy.typing as npt


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

noise_model_name = ['fakecairo', 'fakekolkata', 'fakemontreal']
representations = {}
for noise_name in noise_model_name:
    noise_dict = pd.read_pickle('./NoiseModel/'+noise_name+'.pkl')
    # noise_dict = {}
    # noise_dict['errors'] = []
    # noisy_op = {}
    # noisy_op['gate_qubits'] = [[0, 1]]
    # noisy_op['operations'] = ['cx']
    # noisy_op['probabilities'] = [1/4,1/4,1/4,1/4]
    # noisy_op['instructions'] = []
    # noisy_op_0 = []
    # noisy_op_ins_0 = {}
    # noisy_op_ins_0['name'] = 'pauli'
    # noisy_op_ins_0['params'] = ['II']
    # noisy_op_ins_0['qubits'] = [0, 1]
    # noisy_op_0.append(noisy_op_ins_0)
    # noisy_op_ins_1 = {}
    # noisy_op_ins_1['name'] = 'pauli'
    # noisy_op_ins_1['params'] = ['IX']
    # noisy_op_ins_1['qubits'] = [0, 1]
    # noisy_op_0.append(noisy_op_ins_1)
    # noisy_op_ins_2 = {}
    # noisy_op_ins_2['name'] = 'pauli'
    # noisy_op_ins_2['params'] = ['IY']
    # noisy_op_ins_2['qubits'] = [0, 1]
    # noisy_op_0.append(noisy_op_ins_2)
    # noisy_op_ins_3 = {}
    # noisy_op_ins_3['name'] = 'pauli'
    # noisy_op_ins_3['params'] = ['IZ']
    # noisy_op_ins_3['qubits'] = [0, 1]
    # noisy_op_0.append(noisy_op_ins_3)
    # noisy_op['instructions'].append(noisy_op_0)
    # noise_dict['errors'].append(noisy_op)
    # print("noise_dict loaded")



    single_qubit_noise_operator = []
    # two_qubits_noise_operator = []
    # two_graphs = [] # the map for cx

    for qubit in range(27):
        single_qubit_noise_operator.append([])
        
    for noise in noise_dict['errors']:
        if len(noise['gate_qubits'][0])==1:
            qubit = noise['gate_qubits'][0][0]
            single_qubit_noise_operator[qubit].append(noise)
        # elif len(noise['gate_qubits'][0])==2:
        #     two_graphs.append(noise['gate_qubits'][0])
        #     two_qubits_noise_operator.append(noise)
            
    # print(len(single_qubit_noise_operator))
    # print(len(single_qubit_noise_operator[0]))
    # print(len(two_qubits_noise_operator))

    sigma = 0.125 # for rz
    sigma_cx = 0.25

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
            # single_noisy_super_op_list[qubit].append(single_noisy_super_op_list[qubit][3+i] @ single_noisy_super_op_list[qubit][1]) # rz*sx
            single_noisy_super_op_list[qubit].append(single_noisy_super_op_list[qubit][2] @ single_noisy_super_op_list[qubit][3+i*3]) # x*rz
            single_noisy_super_op_list[qubit].append(single_noisy_super_op_list[qubit][2] @ single_noisy_super_op_list[qubit][3+i*3+1]) # x*ry
            single_noisy_super_op_list[qubit].append(single_noisy_super_op_list[qubit][2] @ single_noisy_super_op_list[qubit][3+i*3+2]) # x*rx
            # single_noisy_super_op_list[qubit].append(single_noisy_super_op_list[qubit][3+i] @ single_noisy_super_op_list[qubit][2]) # rz*x
    single_sup_op_end = time.time()
    print(f"super operators for single operations done. time consumption: {single_sup_op_end-single_sup_op_start}")

            

    # cx_sup_op_start = time.time()
            
    # # cx, cx*sx_0, cx*x_0, cx*id_1, cx*sx_1, cx*x_1, sx_0, x_0, rz's_0, (sx*x)_0, (sx*rz's)_0, (x*rz's)_1, sx_1, x_1, (rz's)_1, (sx*x)_1, (sx*rz's)_1, (x*rz's)_1, 
    # noisy_cx = []
    # two_qubits_noisy_super_op_list = []
    # for index in range(len(two_graphs)):
    #     two_qubits_noisy_super_op_list.append([])
    #     qubit_0 = two_graphs[index][0]
    #     qubit_1 = two_graphs[index][1]
    #     # pdb.set_trace()
        
    #     noisy_operator = two_qubits_noise_operator[index]
    #     ideal_operation_kraus = []
    #     for i in range(len(noisy_operator['operations'])):
    #         kraus_operator = standard_gate_unitary(noisy_operator['operations'][i])
    #         ideal_operation_kraus.append(kraus_operator)


    #     super_op_list = []
    #     prob = noisy_operator['probabilities']
    #     # Special attention!! in qiskit, due to its special order of qubits, the matrix of a union operator should be q1 @ q0
    #     # pdb.set_trace()

    #     for noisy_instruction in noisy_operator['instructions']:
    #         super_op = kraus_to_super(ideal_operation_kraus)
            
    #         for i in range(len(noisy_instruction)):
    #             kraus_set = []
    #             if len(noisy_instruction[i]['qubits'])==2:
    #                 if noisy_instruction[i]['name']=='pauli':
    #                     gate = PauliGate(noisy_instruction[i]['params'][0])
    #                     kraus_set.append(gate.to_matrix())
    #                 elif noisy_instruction[i]['name']=='unitry':
    #                     gate = UnitaryGate(data=noisy_instruction[i]['params'][0])
    #                     kraus_set.append(gate.to_matrix())
    #                 else:
    #                     print('warning!!!!!!!Two qubit error not consider!')

    #             else:
    #                 if(noisy_instruction[i]['qubits'][0]==0):
                        
    #                     if noisy_instruction[i]['name']=='' or noisy_instruction[i]['name']=='measure' or noisy_instruction[i]['name']=='reset':
    #                         continue
                        
    #                     elif noisy_instruction[i]['name']=='kraus':
    #                         for kraus_operator in noisy_instruction[i]['params']:
    #                             kraus_operator = np.kron(np.array([[1,0],[0,1]],dtype=complex),kraus_operator)
    #                             # pdb.set_trace()
    #                             kraus_set.append(kraus_operator)
                                
    #                     elif noisy_instruction[i]['name']=='reset':
    #                         kraus_set.append(np.kron(np.array([[1,0],[0,1]],dtype=complex),np.array([[1,0],[0,0]],dtype=np.complex128)))
    #                         kraus_set.append(np.kron(np.array([[1,0],[0,1]],dtype=complex),np.array([[0,1],[0,0]],dtype=np.complex128)))
    #                     else:
    #                         kraus_operator = np.kron(np.array([[1,0],[0,1]],dtype=complex),standard_gate_unitary(noisy_instruction[i]['name']))
    #                         kraus_set.append(kraus_operator)
                        
    #                 else:
    #                     if noisy_instruction[i]['name']=='' or noisy_instruction[i]['name']=='measure' or noisy_instruction[i]['name']=='reset':
    #                         continue
    #                     elif noisy_instruction[i]['name']=='kraus':
    #                         for kraus_operator in noisy_instruction[i]['params']:
    #                             kraus_operator = np.kron(kraus_operator,np.array([[1,0],[0,1]],dtype=complex)) 
    #                             kraus_set.append(kraus_operator)
                                
    #                     elif noisy_instruction[i]['name']=='reset':
    #                         kraus_set.append(np.kron(np.array([[1,0],[0,0]],dtype=np.complex128),np.array([[1,0],[0,1]],dtype=complex)))
    #                         kraus_set.append(np.kron(np.array([[0,1],[0,0]],dtype=np.complex128),np.array([[1,0],[0,1]],dtype=complex)))
    #                     else:
    #                         kraus_operator = np.kron(standard_gate_unitary(noisy_instruction[i]['name']),np.array([[1,0],[0,1]],dtype=complex))
    #                         kraus_set.append(kraus_operator)
    #             super_op = super_op @ kraus_to_super(kraus_set)
            
    #         super_op_list.append(super_op)
                
    #     noisy_super_op = np.zeros(super_op.shape) 
    #     for i in range(len(super_op_list)):
    #         noisy_super_op = noisy_super_op + prob[i]*super_op_list[i]
    #     two_qubits_noisy_super_op_list[index].append(noisy_super_op) # cx
        
    #     # # sx_0
    #     # two_qubits_noisy_super_op_list[index].append(np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),single_noisy_super_op_list[qubit_0][1]))
        
    #     # # x_0
    #     # two_qubits_noisy_super_op_list[index].append(np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),single_noisy_super_op_list[qubit_0][2]))
        
        
        
    #     # # sx_1
    #     # two_qubits_noisy_super_op_list[index].append(np.kron(single_noisy_super_op_list[qubit_1][1],np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)))
        
    #     # # x_1
    #     # two_qubits_noisy_super_op_list[index].append(np.kron(single_noisy_super_op_list[qubit_1][2],np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)))
        
    #     qubit_op_0 = []
    #     # cx*sx_0
    #     qubit_op_0.append(two_qubits_noisy_super_op_list[index][0] @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),single_noisy_super_op_list[qubit_0][1]))
        
    #     # cx*x_0
    #     qubit_op_0.append(two_qubits_noisy_super_op_list[index][0] @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),single_noisy_super_op_list[qubit_0][2])) 
        
        
    #     # cx*rz_0
    #     for i in range(int(1/sigma_cx)+1):
    #         theta = i*np.pi*sigma_cx
    #         rz = np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128), kraus_to_super([RZGate(theta).to_matrix()]))
    #         qubit_op_0.append(rz)
            
    #     # cx*ry_0
    #     for i in range(1,int(1/sigma_cx)+1):
    #         theta = i*np.pi*sigma_cx
    #         ry = np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),kraus_to_super([RZGate(-np.pi).to_matrix()])) @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),single_noisy_super_op_list[qubit_0][1]) @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),kraus_to_super([RZGate(np.pi-theta).to_matrix()])) @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),single_noisy_super_op_list[qubit_0][1])
    #         qubit_op_0.append(ry)
            
    #     # cx*rx_0
    #     for i in range(1,int(1/sigma_cx)+1):
    #         theta = i*np.pi*sigma_cx
    #         rx = np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),kraus_to_super([RZGate(np.pi/2).to_matrix()])) @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),single_noisy_super_op_list[qubit_0][1]) @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),kraus_to_super([RZGate(np.pi+theta).to_matrix()])) @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),single_noisy_super_op_list[qubit_0][1]) @ np.kron(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128),kraus_to_super([RZGate(np.pi/2).to_matrix()]))
    #         qubit_op_0.append(rx)
        

        
        
    #     #TODO!!!!!!!!!!ry rx中间的rz需要theta+np.pi！！！除了这个文件其他的写的都是错的！！！！
    #     qubit_op_1 = []
    #     # cx*sx_1
    #     qubit_op_1.append(two_qubits_noisy_super_op_list[index][0] @ np.kron(single_noisy_super_op_list[qubit_1][1],np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128))) 
        
    #     # cx*x_1
    #     qubit_op_1.append(two_qubits_noisy_super_op_list[index][0] @ np.kron(single_noisy_super_op_list[qubit_1][2], np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128))) 
        
    #     # cx*rz_1
    #     for i in range(int(1/sigma_cx)+1):
    #         theta = i*np.pi*sigma_cx
    #         rz = np.kron(kraus_to_super([RZGate(theta).to_matrix()]), np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128))
    #         qubit_op_1.append(rz)
            
    #     # cx*ry_1
    #     for i in range(1,int(1/sigma_cx)+1):
    #         theta = i*np.pi*sigma_cx
    #         ry = np.kron(kraus_to_super([RZGate(-np.pi).to_matrix()]),np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(single_noisy_super_op_list[qubit_1][1],np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(kraus_to_super([RZGate(np.pi-theta).to_matrix()]),np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(single_noisy_super_op_list[qubit_1][1],np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128))
    #         qubit_op_1.append(ry)
            
    #     for i in range(1, int(1/sigma_cx)+1):
    #         theta = i*np.pi*sigma_cx
    #         rx = np.kron(kraus_to_super([RZGate(np.pi/2).to_matrix()]),np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(single_noisy_super_op_list[qubit_1][1],np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(kraus_to_super([RZGate(np.pi+theta).to_matrix()]),np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(single_noisy_super_op_list[qubit_1][1],np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128)) @ np.kron(kraus_to_super([RZGate(np.pi/2).to_matrix()]),np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.complex128))
    #         qubit_op_1.append(rx)




    #     for i in range(len(qubit_op_0)):
    #         for j in range(len(qubit_op_1)):
    #             two_qubits_noisy_super_op_list[index].append(qubit_op_0[i]@qubit_op_1[j])
        
        
        
        
    #     noisy_cx.append(two_qubits_noisy_super_op_list[index][0])

    # # t = []
    # # for i in range(len(noisy_cx)):
    # #     t.append(sum(sum(abs(noisy_cx[i]-noisy_cx[0]))))
        
    # # print(t)
            
    # # pdb.set_trace()

    # cx_sup_op_end = time.time()
    # print(f"super operators for cx operations done. time consumption: {cx_sup_op_end-cx_sup_op_start}")


        
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
            
            # basis_noise_circ_rz_sx = QuantumCircuit(27)
            # basis_noise_circ_rz_sx.rz(theta,i)
            # basis_noise_circ_rz_sx.sx(i)
            # basis_noise_circs.append(basis_noise_circ_rz_sx)
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
            # basis_noise_circ_rz_x = QuantumCircuit(27)
            # basis_noise_circ_rz_x.rz(theta,i)
            # basis_noise_circ_rz_x.x(i)
            # basis_noise_circs.append(basis_noise_circ_rz_x)
        
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
        
pdb.set_trace()

        
            
    # cx_rep = []


    # circ_cx_start = time.time()

    # basis_noise_circs = []


    # basis_noise_circ_cx = QuantumCircuit(2)
    # basis_noise_circ_cx.cx(0,1)
    # basis_noise_circs.append(basis_noise_circ_cx)


    # # basis_noise_circ_sx_0 = QuantumCircuit(2)
    # # basis_noise_circ_sx_0.sx(0)
    # # basis_noise_circ_sx_0.rz(0,1)
    # # basis_noise_circs.append(basis_noise_circ_sx_0)

    # # basis_noise_circ_x_0 = QuantumCircuit(2)
    # # basis_noise_circ_x_0.x(0)
    # # basis_noise_circ_x_0.rz(0,1)
    # # basis_noise_circs.append(basis_noise_circ_x_0)

    # # basis_noise_circ_sx_1 = QuantumCircuit(2)
    # # basis_noise_circ_sx_1.sx(1)
    # # basis_noise_circ_sx_1.rz(0,0)
    # # basis_noise_circs.append(basis_noise_circ_sx_1)

    # # basis_noise_circ_x_1 = QuantumCircuit(2)
    # # basis_noise_circ_x_1.x(1)
    # # basis_noise_circ_x_1.rz(0,0)
    # # basis_noise_circs.append(basis_noise_circ_x_1)


    # # with cx
    # # 0sx
    # basis_noise_circ_cx_1sx_0sx = QuantumCircuit(2)
    # basis_noise_circ_cx_1sx_0sx.cx(0, 1)
    # basis_noise_circ_cx_1sx_0sx.sx(0)
    # basis_noise_circ_cx_1sx_0sx.sx(1)
    # basis_noise_circs.append(basis_noise_circ_cx_1sx_0sx)

    # basis_noise_circ_cx_1x_0sx = QuantumCircuit(2)
    # basis_noise_circ_cx_1x_0sx.cx(0, 1)
    # basis_noise_circ_cx_1x_0sx.sx(0)
    # basis_noise_circ_cx_1x_0sx.x(1)
    # basis_noise_circs.append(basis_noise_circ_cx_1x_0sx)

    # for i in range(int(1/sigma_cx)+1):
    #     theta = i*np.pi*sigma_cx
    #     basis_noise_circ_cx_1rz_0sx = QuantumCircuit(2)
    #     basis_noise_circ_cx_1rz_0sx.cx(0,1)
    #     basis_noise_circ_cx_1rz_0sx.sx(0)
    #     basis_noise_circ_cx_1rz_0sx.rz(theta,1)
    #     basis_noise_circs.append(basis_noise_circ_cx_1rz_0sx)
        
    # for i in range(1, int(1/sigma_cx)+1):
    #     theta = i*np.pi*sigma_cx
    #     basis_noise_circ_cx_1ry_0sx = QuantumCircuit(2)
    #     basis_noise_circ_cx_1ry_0sx.cx(0,1)
    #     basis_noise_circ_cx_1ry_0sx.sx(0)
    #     basis_noise_circ_cx_1ry_0sx.rz(-np.pi,1)
    #     basis_noise_circ_cx_1ry_0sx.sx(1)
    #     basis_noise_circ_cx_1ry_0sx.rz(np.pi-theta,1)
    #     basis_noise_circ_cx_1ry_0sx.sx(1)
    #     basis_noise_circs.append(basis_noise_circ_cx_1ry_0sx)

    # for i in range(1, int(1/sigma_cx)+1):
    #     theta = i*np.pi*sigma_cx
    #     basis_noise_circ_cx_1rx_0sx = QuantumCircuit(2)
    #     basis_noise_circ_cx_1rx_0sx.cx(0,1)
    #     basis_noise_circ_cx_1rx_0sx.sx(0)
    #     basis_noise_circ_cx_1rx_0sx.rz(np.pi/2,1)
    #     basis_noise_circ_cx_1rx_0sx.sx(1)
    #     basis_noise_circ_cx_1rx_0sx.rz(np.pi+theta,1)
    #     basis_noise_circ_cx_1rx_0sx.sx(1)
    #     basis_noise_circ_cx_1rx_0sx.rz(np.pi/2,1)
    #     basis_noise_circs.append(basis_noise_circ_cx_1rx_0sx)

    # # 0x
    # basis_noise_circ_cx_1sx_0x = QuantumCircuit(2)
    # basis_noise_circ_cx_1sx_0x.cx(0, 1)
    # basis_noise_circ_cx_1sx_0x.sx(0)
    # basis_noise_circ_cx_1sx_0x.sx(1)
    # basis_noise_circs.append(basis_noise_circ_cx_1sx_0x)

    # basis_noise_circ_cx_1x_0x = QuantumCircuit(2)
    # basis_noise_circ_cx_1x_0x.cx(0, 1)
    # basis_noise_circ_cx_1x_0x.sx(0)
    # basis_noise_circ_cx_1x_0x.x(1)
    # basis_noise_circs.append(basis_noise_circ_cx_1x_0x)

    # for i in range(int(1/sigma_cx)+1):
    #     theta = i*np.pi*sigma_cx
    #     basis_noise_circ_cx_1rz_0x = QuantumCircuit(2)
    #     basis_noise_circ_cx_1rz_0x.cx(0,1)
    #     basis_noise_circ_cx_1rz_0x.x(0)
    #     basis_noise_circ_cx_1rz_0x.rz(theta,1)
    #     basis_noise_circs.append(basis_noise_circ_cx_1rz_0x)
        
    # for i in range(1, int(1/sigma_cx)+1):
    #     theta = i*np.pi*sigma_cx
    #     basis_noise_circ_cx_1ry_0x = QuantumCircuit(2)
    #     basis_noise_circ_cx_1ry_0x.cx(0,1)
    #     basis_noise_circ_cx_1ry_0x.x(0)
    #     basis_noise_circ_cx_1ry_0x.rz(-np.pi,1)
    #     basis_noise_circ_cx_1ry_0x.sx(1)
    #     basis_noise_circ_cx_1ry_0x.rz(np.pi-theta,1)
    #     basis_noise_circ_cx_1ry_0x.sx(1)
    #     basis_noise_circs.append(basis_noise_circ_cx_1ry_0x)

    # for i in range(1, int(1/sigma_cx)+1):
    #     theta = i*np.pi*sigma_cx
    #     basis_noise_circ_cx_1rx_0x = QuantumCircuit(2)
    #     basis_noise_circ_cx_1rx_0x.cx(0,1)
    #     basis_noise_circ_cx_1rx_0x.x(0)
    #     basis_noise_circ_cx_1rx_0x.rz(np.pi/2,1)
    #     basis_noise_circ_cx_1rx_0x.sx(1)
    #     basis_noise_circ_cx_1rx_0x.rz(np.pi+theta,1)
    #     basis_noise_circ_cx_1rx_0x.sx(1)
    #     basis_noise_circ_cx_1rx_0x.rz(np.pi/2,1)
    #     basis_noise_circs.append(basis_noise_circ_cx_1rx_0x)
        
    # # 0rz
    # for i in range(int(1/sigma_cx)+1):
    #     theta_0 = i*np.pi*sigma_cx
    #     basis_noise_circ_cx_1sx_0rz = QuantumCircuit(2)
    #     basis_noise_circ_cx_1sx_0rz.cx(0, 1)
    #     basis_noise_circ_cx_1sx_0rz.rz(theta_0,0)
    #     basis_noise_circ_cx_1sx_0rz.sx(1)
    #     basis_noise_circs.append(basis_noise_circ_cx_1sx_0rz)

    #     basis_noise_circ_cx_1x_0rz = QuantumCircuit(2)
    #     basis_noise_circ_cx_1x_0rz.cx(0, 1)
    #     basis_noise_circ_cx_1x_0rz.rz(theta_0,0)
    #     basis_noise_circ_cx_1x_0rz.x(1)
    #     basis_noise_circs.append(basis_noise_circ_cx_1x_0rz)

    #     for j in range(int(1/sigma_cx)+1):
    #         theta_1 = j*np.pi*sigma_cx
    #         basis_noise_circ_cx_1rz_0rz = QuantumCircuit(2)
    #         basis_noise_circ_cx_1rz_0rz.cx(0,1)
    #         basis_noise_circ_cx_1rz_0rz.rz(theta_0,0)
    #         basis_noise_circ_cx_1rz_0rz.rz(theta_1,1)
    #         basis_noise_circs.append(basis_noise_circ_cx_1rz_0rz)
            
    #     for j in range(1, int(1/sigma_cx)+1):
    #         theta_1 = j*np.pi*sigma_cx
    #         basis_noise_circ_cx_1ry_0rz = QuantumCircuit(2)
    #         basis_noise_circ_cx_1ry_0rz.cx(0,1)
    #         basis_noise_circ_cx_1ry_0rz.rz(theta_0,0)
    #         basis_noise_circ_cx_1ry_0rz.rz(-np.pi,1)
    #         basis_noise_circ_cx_1ry_0rz.sx(1)
    #         basis_noise_circ_cx_1ry_0rz.rz(np.pi-theta_1,1)
    #         basis_noise_circ_cx_1ry_0rz.sx(1)
    #         basis_noise_circs.append(basis_noise_circ_cx_1ry_0rz)

    #     for j in range(1, int(1/sigma_cx)+1):
    #         theta_1 = j*np.pi*sigma_cx
    #         basis_noise_circ_cx_1rx_0rz = QuantumCircuit(2)
    #         basis_noise_circ_cx_1rx_0rz.cx(0,1)
    #         basis_noise_circ_cx_1rx_0rz.rz(theta_0,0)
    #         basis_noise_circ_cx_1rx_0rz.rz(np.pi/2,1)
    #         basis_noise_circ_cx_1rx_0rz.sx(1)
    #         basis_noise_circ_cx_1rx_0rz.rz(np.pi+theta_1,1)
    #         basis_noise_circ_cx_1rx_0rz.sx(1)
    #         basis_noise_circ_cx_1rx_0rz.rz(np.pi/2,1)
    #         basis_noise_circs.append(basis_noise_circ_cx_1rx_0rz)

    # # 0ry
    # for i in range(1, int(1/sigma_cx)+1):
    #     theta_0 = i*np.pi*sigma_cx
    #     basis_noise_circ_cx_1sx_0ry = QuantumCircuit(2)
    #     basis_noise_circ_cx_1sx_0ry.cx(0, 1)
    #     basis_noise_circ_cx_1sx_0ry.rz(-np.pi,0)
    #     basis_noise_circ_cx_1sx_0ry.sx(0)
    #     basis_noise_circ_cx_1sx_0ry.rz(np.pi-theta_0,0)
    #     basis_noise_circ_cx_1sx_0ry.sx(0)
    #     basis_noise_circ_cx_1sx_0ry.sx(1)
    #     basis_noise_circs.append(basis_noise_circ_cx_1sx_0ry)

    #     basis_noise_circ_cx_1x_0ry = QuantumCircuit(2)
    #     basis_noise_circ_cx_1x_0ry.cx(0, 1)
    #     basis_noise_circ_cx_1x_0ry.rz(-np.pi,0)
    #     basis_noise_circ_cx_1x_0ry.sx(0)
    #     basis_noise_circ_cx_1x_0ry.rz(np.pi-theta_0,0)
    #     basis_noise_circ_cx_1x_0ry.sx(0)
    #     basis_noise_circ_cx_1x_0ry.x(1)
    #     basis_noise_circs.append(basis_noise_circ_cx_1x_0ry)

    #     for j in range(int(1/sigma_cx)+1):
    #         theta_1 = j*np.pi*sigma_cx
    #         basis_noise_circ_cx_1rz_0ry = QuantumCircuit(2)
    #         basis_noise_circ_cx_1rz_0ry.cx(0,1)
    #         basis_noise_circ_cx_1rz_0ry.rz(-np.pi,0)
    #         basis_noise_circ_cx_1rz_0ry.sx(0)
    #         basis_noise_circ_cx_1rz_0ry.rz(np.pi-theta_0,0)
    #         basis_noise_circ_cx_1rz_0ry.sx(0)
    #         basis_noise_circ_cx_1rz_0ry.rz(theta_1,1)
    #         basis_noise_circs.append(basis_noise_circ_cx_1rz_0ry)
            
    #     for j in range(1, int(1/sigma_cx)+1):
    #         theta_1 = j*np.pi*sigma_cx
    #         basis_noise_circ_cx_1ry_0ry = QuantumCircuit(2)
    #         basis_noise_circ_cx_1ry_0ry.cx(0,1)
    #         basis_noise_circ_cx_1ry_0ry.rz(-np.pi,0)
    #         basis_noise_circ_cx_1ry_0ry.sx(0)
    #         basis_noise_circ_cx_1ry_0ry.rz(np.pi-theta_0,0)
    #         basis_noise_circ_cx_1ry_0ry.sx(0)
    #         basis_noise_circ_cx_1ry_0ry.rz(-np.pi,1)
    #         basis_noise_circ_cx_1ry_0ry.sx(1)
    #         basis_noise_circ_cx_1ry_0ry.rz(np.pi-theta_1,1)
    #         basis_noise_circ_cx_1ry_0ry.sx(1)
    #         basis_noise_circs.append(basis_noise_circ_cx_1ry_0ry)

    #     for j in range(1, int(1/sigma_cx)+1):
    #         theta_1 = j*np.pi*sigma_cx
    #         basis_noise_circ_cx_1rx_0ry = QuantumCircuit(2)
    #         basis_noise_circ_cx_1rx_0ry.cx(0,1)
    #         basis_noise_circ_cx_1rx_0ry.rz(-np.pi,0)
    #         basis_noise_circ_cx_1rx_0ry.sx(0)
    #         basis_noise_circ_cx_1rx_0ry.rz(np.pi-theta_0,0)
    #         basis_noise_circ_cx_1rx_0ry.sx(0)
    #         basis_noise_circ_cx_1rx_0ry.rz(np.pi/2,1)
    #         basis_noise_circ_cx_1rx_0ry.sx(1)
    #         basis_noise_circ_cx_1rx_0ry.rz(np.pi+theta_1,1)
    #         basis_noise_circ_cx_1rx_0ry.sx(1)
    #         basis_noise_circ_cx_1rx_0ry.rz(np.pi/2,1)
    #         basis_noise_circs.append(basis_noise_circ_cx_1rx_0ry)

    # # 0rx
    # for i in range(1, int(1/sigma_cx)+1):
    #     theta_0 = i*np.pi*sigma_cx
    #     basis_noise_circ_cx_1sx_0rx = QuantumCircuit(2)
    #     basis_noise_circ_cx_1sx_0rx.cx(0, 1)
    #     basis_noise_circ_cx_1sx_0rx.rz(np.pi/2,0)
    #     basis_noise_circ_cx_1sx_0rx.sx(0)
    #     basis_noise_circ_cx_1sx_0rx.rz(np.pi+theta_0,0)
    #     basis_noise_circ_cx_1sx_0rx.sx(0)
    #     basis_noise_circ_cx_1sx_0rx.rz(np.pi/2,0)
    #     basis_noise_circ_cx_1sx_0rx.sx(1)
    #     basis_noise_circs.append(basis_noise_circ_cx_1sx_0rx)

    #     basis_noise_circ_cx_1x_0rx = QuantumCircuit(2)
    #     basis_noise_circ_cx_1x_0rx.cx(0, 1)
    #     basis_noise_circ_cx_1x_0rx.rz(np.pi/2,0)
    #     basis_noise_circ_cx_1x_0rx.sx(0)
    #     basis_noise_circ_cx_1x_0rx.rz(np.pi+theta_0,0)
    #     basis_noise_circ_cx_1x_0rx.sx(0)
    #     basis_noise_circ_cx_1x_0rx.rz(np.pi/2,0)
    #     basis_noise_circ_cx_1x_0rx.x(1)
    #     basis_noise_circs.append(basis_noise_circ_cx_1x_0rx)

    #     for j in range(int(1/sigma_cx)+1):
    #         theta_1 = j*np.pi*sigma_cx
    #         basis_noise_circ_cx_1rz_0rx = QuantumCircuit(2)
    #         basis_noise_circ_cx_1rz_0rx.cx(0,1)
    #         basis_noise_circ_cx_1rz_0rx.rz(np.pi/2,0)
    #         basis_noise_circ_cx_1rz_0rx.sx(0)
    #         basis_noise_circ_cx_1rz_0rx.rz(np.pi+theta_0,0)
    #         basis_noise_circ_cx_1rz_0rx.sx(0)
    #         basis_noise_circ_cx_1rz_0rx.rz(np.pi/2,0)
    #         basis_noise_circ_cx_1rz_0rx.rz(theta_1,1)
    #         basis_noise_circs.append(basis_noise_circ_cx_1rz_0rx)
            
    #     for j in range(1, int(1/sigma_cx)+1):
    #         theta_1 = j*np.pi*sigma_cx
    #         basis_noise_circ_cx_1ry_0rx = QuantumCircuit(2)
    #         basis_noise_circ_cx_1ry_0rx.cx(0,1)
    #         basis_noise_circ_cx_1ry_0rx.rz(np.pi/2,0)
    #         basis_noise_circ_cx_1ry_0rx.sx(0)
    #         basis_noise_circ_cx_1ry_0rx.rz(np.pi+theta_0,0)
    #         basis_noise_circ_cx_1ry_0rx.sx(0)
    #         basis_noise_circ_cx_1ry_0rx.rz(np.pi/2,0)
    #         basis_noise_circ_cx_1ry_0rx.rz(-np.pi,1)
    #         basis_noise_circ_cx_1ry_0rx.sx(1)
    #         basis_noise_circ_cx_1ry_0rx.rz(np.pi-theta_1,1)
    #         basis_noise_circ_cx_1ry_0rx.sx(1)
    #         basis_noise_circs.append(basis_noise_circ_cx_1ry_0rx)

    #     for j in range(1, int(1/sigma_cx)+1):
    #         theta_1 = j*np.pi*sigma_cx
    #         basis_noise_circ_cx_1rx_0rx = QuantumCircuit(2)
    #         basis_noise_circ_cx_1rx_0rx.cx(0,1)
    #         basis_noise_circ_cx_1rx_0rx.rz(np.pi/2,0)
    #         basis_noise_circ_cx_1rx_0rx.sx(0)
    #         basis_noise_circ_cx_1rx_0rx.rz(np.pi+theta_0,0)
    #         basis_noise_circ_cx_1rx_0rx.sx(0)
    #         basis_noise_circ_cx_1rx_0rx.rz(np.pi/2,0)
    #         basis_noise_circ_cx_1rx_0rx.rz(np.pi/2,1)
    #         basis_noise_circ_cx_1rx_0rx.sx(1)
    #         basis_noise_circ_cx_1rx_0rx.rz(np.pi+theta_1,1)
    #         basis_noise_circ_cx_1rx_0rx.sx(1)
    #         basis_noise_circ_cx_1rx_0rx.rz(np.pi/2,1)
    #         basis_noise_circs.append(basis_noise_circ_cx_1rx_0rx)

        
    # circ_cx_end = time.time()
    # print(f"circuits constructed for cx. time consumption: {circ_cx_end-circ_cx_start}")
    # print(f"{len(basis_noise_circs)} noisy basis circuits in total")


    # cx_rep_start = time.time()

    # for i in range(len(two_graphs)):
    #     cx_rep_i_start = time.time()
    #     basis_noise_circs_27 = []
        
    #     circuit_i_start = time.time()
    #     for j in range(len(basis_noise_circs)):
    #         qc_27 = QuantumCircuit(27)
    #         qc_27.append(basis_noise_circs[j], [two_graphs[i][0],two_graphs[i][1]])
    #         basis_noise_circs_27.append(qc_27.decompose())
    #     circuit_i_end = time.time()
    #     print(f"circuits constructed for the {i}'s pair of cx. time consumption: {circuit_i_end-circuit_i_start}")

    #     NoisyOperations_i_start = time.time()
    #     noisy_operations = [
    #         pec.NoisyOperation(circuit=c, channel_matrix=m)
    #         for c,m in zip(basis_noise_circs_27, two_qubits_noisy_super_op_list[i])
    #     ]
    #     ideal_cx = QuantumCircuit(27)
    #     ideal_cx.cx(two_graphs[i][1],two_graphs[i][0])
    #     ideal_cirq_circuit, _ = convert_to_mitiq(ideal_cx)
    #     # t = kraus_to_super(cast(List[npt.NDArray[np.complex64]], kraus(ideal_cirq_circuit)))
    #     # print(t)
    #     # print(two_qubits_noisy_super_op_list[i][0] * abs(two_qubits_noisy_super_op_list[i][0]>1e-4))
    #     # pdb.set_trace()
    #     # print(sum(sum(abs(t-two_qubits_noisy_super_op_list[i][0]))))
    #     NoisyOperations_i_end = time.time()
    #     print(f"NoisyOperation constructed for the {i}'s pair of cx. time consumption: {NoisyOperations_i_end-NoisyOperations_i_start}")
        
        
        
    #     init_guess = np.zeros(len(two_qubits_noisy_super_op_list[i]))
    #     init_guess[0] = 0.7
    #     init_guess[1] = 0.1
    #     init_guess[2] = 0.2
    #     rep = find_optimal_representation(ideal_cx, noisy_operations, tol=1e-5, is_qubit_dependent=True, initial_guess = init_guess)
    #     # print(f'cx_rep for num {i} qubits {two_graphs[i][0]},{two_graphs[i][1]} is\n\n{rep}')
    #     print('\n\n')
    #     cx_rep.append(rep)
    #     cx_rep_i_end = time.time()
    #     print(f"representation for the {i}'s pair found. time consumption: {cx_rep_i_end-cx_rep_i_start}")

    #     #TODO: switch back 0 and 1


    # cx_rep_end = time.time()
    # print(f"representations for all cx operation found. time consumption: {cx_rep_end-cx_rep_start}")


        
#TODO: read out error
    