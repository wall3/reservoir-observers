import numpy as np
# from numpy import linalg as LA
import matplotlib.pyplot as plt
# import networkx as nx
# from scipy import integrate

from dynamic_systems import Dynamics
import reservoir


NODES = 400
TIME_STEP = 0.1
TRAINING_TIME = 260
TRANSIENT_TIME = 200
TEST_TIME = 100
INPUTS = [0]
OUTPUTS = [1, 2]
INITIAL_CONDITION = (0.0, 0.0, 0.0)
# PARAMETER CONSTANTS
D = 20 #the average degree of a reservoir node
SPECTRAL_RADIUS = 1.0
SIGMA = 1.0
LEAKAGE_RATE = 1.0
BIAS_CONSTANT = 1.0
BETA = 0.00005

total_time = TRANSIENT_TIME + TRAINING_TIME + TEST_TIME
# options for dynamic systems: rossler, lorenz, modified_lorenz.
dynamic_system = Dynamics.rossler

DATA = reservoir.gen_data(dynamic_system, INITIAL_CONDITION, total_time, TIME_STEP)
input_train, output_train, input_test, output_test = reservoir.split_data(DATA, INPUTS, OUTPUTS, TRAINING_TIME, TRANSIENT_TIME, TIME_STEP)
A = reservoir.gen_A(NODES, D, SPECTRAL_RADIUS)
W_in = reservoir.gen_Win(NODES, len(INPUTS), SIGMA)
W_out, R, rbar, sbar = reservoir.train(input_train, output_train, A, W_in, NODES, len(INPUTS), len(OUTPUTS), TRAINING_TIME, LEAKAGE_RATE, BIAS_CONSTANT, BETA, TIME_STEP)
s_t = reservoir.test(input_test, W_out, A, W_in, NODES, len(INPUTS), len(OUTPUTS), LEAKAGE_RATE, BIAS_CONSTANT, R, rbar, sbar)

print("RMS error in calculation of y: ", np.sqrt(np.mean((s_t[0,:]-output_test[:1000, 0])**2)))
print("RMS error in calculation of z: ", np.sqrt(np.mean((s_t[1,:]-output_test[:1000, 1])**2)))


# This block of code is used to test the variability of the ridge regression parameter.
# betas = [0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
# RMSy = np.zeros(12)
# RMSz = np.zeros(12)
# for idx, BETA in enumerate(betas):
#     W_out, R, rbar, sbar = reservoir.train(input_train, output_train, A, W_in, NODES, len(INPUTS), len(OUTPUTS), TRAINING_TIME, LEAKAGE_RATE, BIAS_CONSTANT, BETA, TIME_STEP)
#     s_t = reservoir.test(input_test, W_out, A, W_in, NODES, len(INPUTS), len(OUTPUTS), LEAKAGE_RATE, BIAS_CONSTANT, R, rbar, sbar)
#     RMSy[idx] = np.sqrt(np.mean((s_t[0,:]-output_test[:1000, 0])**2))
#     RMSz[idx] = np.sqrt(np.mean((s_t[1,:]-output_test[:1000, 1])**2))
#
# print(RMSy)
# print(RMSz)
#
# fig = plt.figure()
# plt.subplot(2,1,1)
# plt.loglog(betas, RMSy)
# plt.title("RMSE in the y-component")
# plt.ylabel('RMS')
#
# plt.subplot(2,1,2)
# plt.loglog(betas, RMSz)
# plt.title("RMSE in the z-component")
# plt.ylabel('RMS')
# plt.xlabel('beta')
#
# plt.show()

# Visualize
t = np.arange(0, TEST_TIME, TIME_STEP)

fig = plt.figure(figsize=(30,20))
plt.subplot(3, 1, 1)
plt.plot(t, input_test[:1000])
plt.ylabel('Input x')

plt.subplot(3, 1, 2)
plt.plot(t, s_t[0,:], 'r', label='Reservoir output')
plt.plot(t, output_test[:1000, 0], 'b--', label='Actual y')
plt.ylabel('output and actual y')
plt.legend(loc='upper right')

plt.subplot(3, 1, 3)
plt.plot(t, s_t[1,:], 'r', label='Reservoir output')
plt.plot(t, output_test[:1000, 1], 'b--', label='Actual z')
plt.xlabel('time (s)')
plt.ylabel('output and actual z')
plt.legend(loc='upper right')

plt.show()
