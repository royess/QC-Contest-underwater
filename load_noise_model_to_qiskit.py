import qiskit
from qiskit.providers.aer.noise import NoiseModel
noise_dict = {
    'errors': [
        {
            'type': 'depolarizing',
            'operations': ['u1', 'u2', 'u3', 'cx'],
            'probabilities': [0.001, 0.002, 0.003, 0.004]
        }
    ]
}

# Load the noise model from the dictionary
noise_model = NoiseModel.from_dict(noise_dict)
print('1')