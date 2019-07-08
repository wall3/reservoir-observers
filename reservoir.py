import numpy as np
from scipy import integrate
import networkx as nx
from numpy import linalg as LA


def gen_data(system, initial_conditions, total_time, time_step):
    #initial_conditions = (0.0, 0.0, 0.0)
    t = np.arange(0, total_time, time_step)
    X = integrate.odeint(system, initial_conditions, t)
    # Preprocess Data: zero mean + unit variance
    for i in range(X.shape[1]):
        X[:,i] -= np.mean(X[:,i])
        X[:,i] = X[:,i]/np.std(X[:,i])
    return X

def split_data(X, inputs, outputs, train_size, transient_time, time_step):
    # In our case x-coord is the input
    u_t = X[:,inputs]
    # y and z-coord are output
    s_t = X[:,outputs]
    idx = int((train_size + transient_time)/time_step)
    temp = np.split(u_t, [idx])
    X_train = temp[0]
    X_test = temp[1]
    temp = np.split(s_t, [idx])
    Y_train = temp[0]
    Y_test = temp[1]

    return X_train, Y_train, X_test, Y_test

def gen_A(N, D, rho=1.0):
    """This function creates the adjacency matrix for the reservoir layer.
    Creates a random (NxN) matrix through the Erdos-Renyi graph algorithm.

    Keyword arguments:
    N -- the number of nodes
    D -- the average degree of a reservoir node
    rho -- the spectral radius of A (default 1.0)
    """
    A = nx.erdos_renyi_graph(N, D/N, directed=True) #, seed=1)
    A = nx.to_numpy_matrix(A)
    nz_idx = np.nonzero(A)
    nz_val = np.count_nonzero(A)
    #np.random.seed(1) # Delete this line afterwards
    elems = np.random.uniform(-1, 1, nz_val)
    for i in range(nz_val):
        A[nz_idx[0][i], nz_idx[1][i]] = elems[i]
    eigVals, _ = LA.eig(A)
    maxEigVal = np.max(eigVals)
    A = (rho/maxEigVal)*A
    return A

def gen_Win(N, M, sigma=1.0):
    """This function creates the W_in

    Keyword arguments:
    N -- the number of nodes
    M -- the number of inputs
    sigma -- (default 1.0)
    """
    W_in = np.zeros((N, M))
    for i in range(N):
        idx = np.random.randint(0, M)
        W_in[i, idx] = np.random.uniform(-sigma, sigma)
    #print("W_in shape: ", W_in.shape)
    return W_in

def train(u_t, s_t, A, W_in, N, M, P, training_time, alpha, bias, beta, time_step):
    """This function is used to train the reservoir layer and output W_out

    Keyword arguments:
    u_t -- the input
    s_t -- the output
    A -- the adjacency matrix
    W_in --
    N -- number of nodes in reservoir
    M -- numper of inputs
    P -- number of outputs
    transient_time --
    alpha -- leakage rate
    bias -- bias constant
    beta -- ridge-regression parameter
    """
    K = u_t.shape[0]
    training_length = int(training_time/time_step)
    R = np.zeros((N, training_length))
    S = np.zeros((P, training_length))
    temp_r = np.vstack(np.zeros(N))
    for t in range(K):
        temp_r = (1-alpha) * temp_r + alpha * np.tanh((np.dot(A, temp_r)) + np.dot(W_in, np.vstack(u_t[t])) + bias)
        print(temp_r.shape)
        if t > K - training_length:
            k = t - K + training_length
            R[:,k] = temp_r.flatten()
            S[:,k] = s_t[t,:]
    rbar = (1/training_length)*np.sum(R, axis=1)
    sbar = (1/training_length)*np.sum(S, axis=1)
    dR = R - np.vstack(rbar)
    dS = S - np.vstack(sbar)
    W_out = np.dot(dS, dR.T) @ LA.inv(np.dot(dR, dR.T) + beta * np.identity(N))

    return W_out, R, rbar, sbar

def test(u_t, W_out, A, W_in, N, M, P, alpha, bias, R, rbar, sbar):
    """This function is used to test the reservoir and output s_t
    on the given u_t

    Keyword arguments:
    u_t -- the input
    W_out --
    A -- the adjacency matrix
    W_in --
    N -- number of nodes in reservoir
    M -- numper of inputs
    P -- number of outputs
    alpha -- leakage rate
    bias -- bias constant
    R -- the reservoir states
    rbar --
    sbar --
    """
    test_length = u_t.shape[0]
    temp_r = np.vstack(R[:,-1])
    output = np.zeros(P)
    states = np.zeros((N, test_length))
    s_t = np.zeros((P, test_length))
    for t in range(test_length):
        temp_r = (1-alpha) * temp_r + alpha * np.tanh(np.dot(A, temp_r) + np.dot(W_in, np.vstack(u_t[t])) + bias)
        output = np.dot(W_out, temp_r) - np.vstack(np.dot(W_out, rbar) - sbar)
        states[:,t] = temp_r.flatten()
        s_t[:,t] = output.flatten()
    return s_t
