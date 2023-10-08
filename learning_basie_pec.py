import qiskit
import numpy as np
import pandas as pd
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors.standard_errors import (
    depolarizing_error,
)
from mitiq import Observable, PauliString
from mitiq.pec.representations.learning import (
    learn_depolarizing_noise_parameter,
    learn_biased_noise_parameters,
    
)

from mitiq.pec.representations import (
    represent_operation_with_local_biased_noise,
)

from mitiq.interface.mitiq_qiskit.qiskit_utils import execute_with_noise, initialized_depolarizing_noise

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
from mitiq.pec.representations.depolarizing import represent_operations_in_circuit_with_local_depolarizing_noise

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
from qiskit.circuit.library import RZGate, SXGate, CXGate
from qiskit.circuit.library.generalized_gates import PauliGate
from qiskit.extensions import UnitaryGate
import time
import copy
import pdb















noise_dict = pd.read_pickle('./NoiseModel/fakecairo.pkl')
    
    
# noise_dict_mod_1 = {}
# noise_dict_mod_1['errors'] = []         
# for i in noise_dict['errors']:
#     if(len(i['gate_qubits'][0])==1):
#         if(i['gate_qubits'][0][0] == 6):
#             error_mod = i
#             noise_dict_mod_1['errors'].append(error_mod)
#         elif(i['gate_qubits'][0][0] == 7):
#             error_mod = i
#             noise_dict_mod_1['errors'].append(error_mod)
#     elif(len(i['gate_qubits'][0])==2 and i['gate_qubits'][0][0] == 6 and i['gate_qubits'][0][1] == 7):
#         error_mod = i

#         noise_dict_mod_1['errors'].append(error_mod)  


#  TODO: 后面都需要改成深拷贝
noise_dict_mod = {}
noise_dict_mod['errors'] = []

for i in noise_dict['errors']:
    if(len(i['gate_qubits'][0])==1):
        if(i['gate_qubits'][0][0] == 12):
            error_mod = i
            error_mod['gate_qubits'][0] = (0,) # some detail needed discussion 0 or 1
            noise_dict_mod['errors'].append(error_mod)
        elif(i['gate_qubits'][0][0] == 13):
            error_mod = i
            error_mod['gate_qubits'][0] = (1,)
            noise_dict_mod['errors'].append(error_mod)
        elif(len(i['gate_qubits'][0])==2 and i['gate_qubits'][0][0] == 12 and i['gate_qubits'][0][1] == 13):
            error_mod = i
            error_mod['gate_qubits'][0] = (0,1)
            error_mod['gate_qubits'][0]
            noise_dict_mod['errors'].append(error_mod)


       
            
            

# Load the noise model from the dictionary
# noise_model_1 = NoiseModel.from_dict(noise_dict_mod_1)
noise_model = NoiseModel.from_dict(noise_dict_mod)
print(f'basis_gates: {noise_model.basis_gates}')

circuit = qiskit.QuantumCircuit(2)
circuit.rx(1.14* np.pi, 1)
circuit.rz(0, 0)
circuit.cx(1, 0) # 1,0 or 7,6?
circuit.rx(1.71 * np.pi, 1)
circuit.rx(1.14 * np.pi, 0)

observable = Observable(PauliString("XZ"), PauliString("YY")).matrix() # mitiq内部会自动失去qubitnum信息   what observables need?
# not sure in execute_with_noise, turns into q0 and q1

# set up ideal simulator
def ideal_execute(circuit):
    """Simulate (training) circuits without noise"""
    circuit_copy = circuit.copy()
    noise_model = initialized_depolarizing_noise(0.0)
    return execute_with_noise(circuit_copy, observable, noise_model)


epsilon = 0.05
eta = 0
# simulate biased noise occurs on the CNOT gates
def noisy_execute(circuit):
    
    circuit_copy = circuit.copy()
    # if(circuit_copy.num_qubits==2):
    #     qc_27.append(circuit_copy, [7,6]) 
    #     # trying to regain qubit num info  but unrealistic! cause backend need density matrix. By no means can we make a 2**27 size
    #     # all we can do now is to modify noise model
    #     circuit_copy_re = qc_27
    #     pdb.set_trace()
    # else:
    #     circuit_copy_re = circuit_copy
        
    # noise_model.add_all_qubit_quantum_error(depolarizing_error(epsilon, 2), ["cx"])
    return execute_with_noise(circuit_copy, observable, noise_model)


operations_to_learn = qiskit.QuantumCircuit(2)
operations_to_learn.cx(1, 0) # somehow here uses 1, 0 instead of 0, 1 !! this can help preserving the conversion from qiskit to mitiq

# how to specify the qubit which cx acts on?


[success, epsilon_opt, eta_opt] = learn_biased_noise_parameters(
    operations_to_learn=[operations_to_learn],  # learn rep of CNOT
    
    circuit=circuit,
    ideal_executor=Executor(ideal_execute),
    noisy_executor=Executor(noisy_execute),
    pec_kwargs={"num_samples": 5, "random_state": 1},
    num_training_circuits=5,
    fraction_non_clifford=0.2,
    training_random_state=np.random.RandomState(1),
    epsilon0=1.01 * 0.05,  # initial guess for noise strength
    eta0= eta + 0.01,  # initial guess for noise bias
)
pdb.set_trace()
representations = represent_operation_with_local_biased_noise(
    operations_to_learn, epsilon_opt, eta_opt
)
pdb.set_trace()
