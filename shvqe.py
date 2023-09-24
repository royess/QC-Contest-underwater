"""
DQAS-style optimization for hybrid quantum-Clifford type circuit

Modified from: https://github.com/tencent-quantum-lab/tensorcircuit/blob/master/examples/clifford_optimization.py
"""

import numpy as np
import tensorflow as tf

import tensorcircuit as tc
# import tensorcircuit.quantum as qu
from scipy.sparse import load_npz

import pickle

ctype, rtype = tc.set_dtype("complex64")
K = tc.set_backend("tensorflow")

n = 12 # the number of qubits (must be even for consistency later)
ncz = 3 # number of cz layers in Schrodinger circuit
nlayersq = ncz + 3 # Shrodinger parameter layers + Heisenberg last layer

# training setup
epochs = 100
batch = 1024

# fix random seed
seed = 42
K.set_random_state(seed)
np.random.seed(seed)
tf.keras.utils.set_random_seed(seed)

# dynamical GPU memory allocation
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)
tf.compat.v1.keras.backend.set_session(sess) # The name tf.keras.backend.set_session is deprecated. Please use tf.compat.v1.keras.backend.set_session instead.

# Uncomment to directly generate Hamiltonian
# def read_paul_string_sum_from_file(path):
#     with open(path, 'r') as file:
#         pauli_text_lines = file.readlines()
#     l = []
#     weight = []
#     for line in pauli_text_lines:
#         line = line.replace(' ', '')
#         coeff, pauli_text_string = line.split("*")
#         coeff = float(coeff)
#         weight.append(coeff)
#         ps = []
#         for c in pauli_text_string:
#             if c == 'I':
#                 ps.append(0)
#             elif c == 'X':
#                 ps.append(1)
#             elif c == 'Y':
#                 ps.append(2)
#             elif c == 'Z':
#                 ps.append(3)
#         l.append(ps)
#     return l, weight

# def remove_small_elements(sparse_tensor, thres=1e-6):
#     # Extract the values, indices, and shape from the SparseTensor
#     # TODO: maybe propose as an issue for tc
#     # because their sparse matrix from pauli has lots of small values which should be 0
#     values = sparse_tensor.values
#     indices = sparse_tensor.indices
#     shape = sparse_tensor.dense_shape

#     # Find the indices of elements with absolute value >= thres
#     mask = tf.abs(values) >= thres
#     valid_indices = tf.boolean_mask(indices, mask)
#     valid_values = tf.boolean_mask(values, mask)

#     # Create a new SparseTensor with the valid values and indices
#     new_sparse_tensor = tf.SparseTensor(valid_indices, valid_values, shape)

#     return new_sparse_tensor

# ham_coo = qu.PauliStringSum2COO(*read_paul_string_sum_from_file('Hamiltonian/OHhamiltonian.txt'))
# ham_coo = remove_small_elements(ham_coo)

ham_coo_numpy = load_npz('Hamiltonian/OHham.npz')
ham_coo_indices = np.mat([ham_coo_numpy.row, ham_coo_numpy.col]).transpose()
ham_coo = tf.SparseTensor(ham_coo_indices, ham_coo_numpy.data, ham_coo_numpy.shape)

def hybrid_ansatz(structure, paramq, preprocess="direct", train=True):
    """_summary_

    Parameters
    ----------
    structure : K.Tensor, (n//2, 2)
        parameters to decide graph structure of Clifford circuits
    paramq : K.Tensor, (nlayersq, n, 3)
        parameters in quantum variational circuits, the last layer for Heisenberg circuits
    preprocess : str, optional
        preprocess, by default "direct"

    Returns
    -------
    K.Tensor, [1,]
        loss value
    """
    c = tc.Circuit(n)
    if preprocess == "softmax":
        structure = K.softmax(structure, axis=-1)
    elif preprocess == "most":
        structure = K.onehot(K.argmax(structure, axis=-1), num=2)
    elif preprocess == "direct":
        pass

    structure = K.cast(structure, ctype)
    structure = tf.reshape(structure, shape=[n//2, 2])

    # quantum variational in Schrodinger part, first consider a ring topol
    for j in range(nlayersq-1):
        if j !=0 and j!=nlayersq-2:
            for i in range(j%2,n-1,2):
                c.cz(i, (i+1)%n)
        for i in range(n):
            c.rx(i, theta=paramq[j, i, 0])
            c.ry(i, theta=paramq[j, i, 1])
            c.rz(i, theta=paramq[j, i, 2])

    # Clifford part, which is actually virtual
    if train:
        for j in range(0,n//2-1):
            dis = j + 1
            for i in range(0,n):
                c.unitary(
                    i,
                    (i+dis) % n,
                    unitary=structure[j, 0] * tc.gates.ii().tensor
                    + structure[j, 1] * tc.gates.cz().tensor,
                )

        for i in range(0,n//2):
            c.unitary(
                i,
                i + n//2,
                unitary=structure[n//2-1, 0] * tc.gates.ii().tensor
                + structure[n//2-1, 1] * tc.gates.cz().tensor,
            )
    else: # if not for training, we just put nontrivial gates
        for j in range(0,n//2-1):
            dis = j + 1
            for i in range(0,n):
                if structure[j, 1]==1:
                    c.cz(i, (i+dis) % n)

        for i in range(0,n//2):
            if structure[j, 1]==1:
                c.cz(i, i + n//2)

    for i in range(n):
        c.rx(i, theta=paramq[-1, i, 0])
        c.ry(i, theta=paramq[-1, i, 1])
        c.rz(i, theta=paramq[-1, i, 2])

    return c

def hybrid_vqe(structure, paramq, preprocess="direct"):
    """_summary_

    Parameters
    ----------
    structure : K.Tensor, (n//2, 2)
        parameters to decide graph structure of Clifford circuits
    paramq : K.Tensor, (nlayersq, n, 3)
        parameters in quantum variational circuits, the last layer for Heisenberg circuits
    preprocess : str, optional
        preprocess, by default "direct"

    Returns
    -------
    K.Tensor, [1,]
        loss value
    """
    c = hybrid_ansatz(structure, paramq, preprocess)
    return tc.templates.measurements.operator_expectation(c, ham_coo)

def sampling_from_structure(structures, batch=1):
    ch = structures.shape[-1]
    prob = K.softmax(K.real(structures), axis=-1)
    prob = K.reshape(prob, [-1, ch])
    p = prob.shape[0]
    r = np.stack(
        np.array(
            [np.random.choice(ch, p=K.numpy(prob[i]), size=[batch]) for i in range(p)]
        )
    )
    return r.transpose()


@K.jit
def best_from_structure(structures):
    return K.argmax(structures, axis=-1)


def nmf_gradient(structures, oh):
    """ compute the Monte Carlo gradient with respect of naive mean-field probabilistic model

    Parameters
    ----------
    structures : K.Tensor, (n//2, ch)
        structure parameter for single- or two-qubit gates
    oh : K.Tensor, (n//2, ch), onehot
        a given structure sampled via strcuture parameters (in main function)

    Returns
    -------
    K.Tensor, (n//2 * 2, ch) == (n, ch)
        MC gradients
    """
    choice = K.argmax(oh, axis=-1)
    prob = K.softmax(K.real(structures), axis=-1)
    indices = K.transpose(
        K.stack([K.cast(tf.range(structures.shape[0]), "int64"), choice])
    )
    prob = tf.gather_nd(prob, indices)
    prob = K.reshape(prob, [-1, 1])
    prob = K.tile(prob, [1, structures.shape[-1]])

    return K.real(
        tf.tensor_scatter_nd_add(
            tf.cast(-prob, dtype=ctype),
            indices,
            tf.ones([structures.shape[0]], dtype=ctype),
        )
    ) # in oh : 1-p, not in oh : -p

# vmap for a batch of structures
nmf_gradient_vmap = K.jit(
    K.vmap(nmf_gradient, vectorized_argnums=1))

# vvag for a batch of structures
vvag_hybrid = K.jit(
    K.vectorized_value_and_grad(hybrid_vqe, vectorized_argnums=(0,), argnums=(1,)),
    static_argnums=(2,))

# vag_quantum = K.jit(
#     K.value_and_grad(quantum_ansatz)
# )

def train_hybrid(stddev=0.05, lr=None, epochs=2000, debug_step=50, batch=256, verbose=False):
    params = K.implicit_randn([n//2, 2], stddev=stddev)
    paramq = K.implicit_randn([nlayersq, n, 3], stddev=stddev) * 2*np.pi
    if lr is None:
        lr = tf.keras.optimizers.schedules.ExponentialDecay(0.06, 1000, 0.5)
    structure_opt = K.optimizer(tf.keras.optimizers.Adam(lr))

    avcost = 0
    avcost2 = 0
    loss_history = []
    for epoch in range(epochs):  # iteration to update strcuture param
        # random sample some structures
        batched_stucture = K.onehot(
            sampling_from_structure(params, batch=batch),
            num=params.shape[-1],
        )
        vs, gq = vvag_hybrid(batched_stucture, paramq, "direct")
        loss_history.append(np.min(vs))
        gq = gq[0]
        avcost = K.mean(vs) # average cost of the batch
        gs = nmf_gradient_vmap(params, batched_stucture)  # \nabla lnp
        gs = K.mean(K.reshape(vs - avcost2, [-1, 1, 1]) * gs, axis=0)
        # avcost2 is averaged cost in the last epoch
        avcost2 = avcost

        [params, paramq] = structure_opt.update([gs, gq], [params, paramq])
        if epoch % debug_step == 0 or epoch == epochs - 1:
            print("----------epoch %s-----------" % epoch)
            print(
                "batched average loss: ",
                np.mean(vs),
                "minimum candidate loss: ",
                np.min(vs),
            )

            # max over choices, min over layers and qubits
            minp = tf.math.reduce_min(tf.math.reduce_max(tf.math.softmax(params), axis=-1))
            if minp > 0.5:
                print("probability converged")

            if verbose:
                print(
                    "strcuture parameter: \n",
                    params.numpy()
                )

            cand_preset = best_from_structure(params)
            print(cand_preset)
            print("current recommendation loss: ", hybrid_vqe(params, paramq, "most"))

    loss_history = np.array(loss_history)
    return hybrid_vqe(params, paramq, "most"), params, paramq, loss_history


if __name__ == "__main__":
    print('Train hybrid.')
    ee, params, paramq, loss_history = train_hybrid(epochs=epochs, batch=batch, verbose=True)
    np.save('saved_models/hybrid_params.npy', params.numpy())
    np.save('saved_models/hybrid_paramq.npy', paramq.numpy())
    np.savetxt('saved_models/hybrid_loss_history.txt', loss_history)
    print('Energy:', ee)

    ref_value = -74.38714627
    nuclear_repulsion_energy = 4.365374966545
    result = ee + nuclear_repulsion_energy
    error_rate = abs(abs(ref_value - result) / ref_value * 100)
    print('Ref energy:', ref_value)
    print('Error rate:', error_rate)

    c = hybrid_ansatz(params, paramq, "most", train=False) # output the compact circuit
    print(c.draw())
    with open('saved_models/hybrid.qasm', 'w+') as file:
        file.write(c.to_openqasm())
