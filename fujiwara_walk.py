import numpy as np
import matplotlib.pyplot as plt

# Implementation of Fujiwara's walk with a position-dependent coin operator

'''
Input : number of nodes
Output : matrix S allowing us to modify the position of the walker
'''
def get_s(n):
    s0 = np.zeros((n,n))
    s1 = np.zeros((n,n))
    for i in range(n):
        s0[(i-1) % n][i] = 1
        s1[(i+1) % n][i] = 1
    s = np.kron(s0, [[1,0],[0,0]]) + np.kron(s1, [[0,0],[0,1]])
    return s

'''
Inputs : 
    n : number of nodes
    coin : coin operator
Output : matrix U applying the coin operator on the coin space and the identity on the position space
'''
def get_u(n, coin):
    return np.kron(np.identity(n), coin)

'''
Inputs:
    vector : state vector
    n : number of nodes
    coin : coin operator
    n_step : number of steps
    
    Optionals arguments :
        position_dependent : True if we want a position-dependent coin operator
        coin_list : In case we want a position-dependent coin operator, coin_list[i] contains the coin operator
                    we want to apply on position i
Output : the state vector of the walker after n_step steps
'''
def quantum_walk(vector, n, coin, n_step, position_dependent = False, coin_list = []):
    s = get_s(n)
    if(position_dependent):
        u = get_u_position_dependent(n, coin_list)
    else:
        u = get_u(n, coin)
    q = np.matmul(s,u)
    f = np.linalg.matrix_power(q, n_step)
    return np.matmul(f, vector)

'''
Inputs :
    vector : state vector
    n : number of nodes
Output : dictionnary containing the probability distribution of the positions
'''
def decode(vector, n):
    prob_distrib = {}
    for i in range(n):
        if(vector[2*i][0] != 0):
            prob_distrib[i] = np.abs(vector[2*i][0])**2
        if(vector[2*i + 1][0] != 0):
            if i in prob_distrib.keys():
                prob_distrib[i] += np.abs(vector[2*i +1][0])**2
            else:
                prob_distrib[i] = np.abs(vector[2*i +1][0])**2
    return prob_distrib

'''
Input :
    vector : state vector
Output : the state vector normalized
'''
def normalize(vector):
    norm = 0
    for i in vector:
        norm += np.abs(i[0])**2
    if(norm != 0):
        vector /= np.sqrt(norm)
    return vector

'''
Inputs :
    n : number of nodes
    pos_init : amplitudes list of the position
    coin_init : amplitudes list of the coin, [1,1] corresponding to the state (|0>+|1>)/sqrt(2)
Output : the state vector normalized
'''
def init(n, pos_init, coin_init):
    vector = np.zeros((2*n, 1))
    coin_len = len(coin_init)
    for i in range(len(pos_init)):
        for j in range(coin_len):
            vector[(coin_len*i)+j] = pos_init[i] * coin_init[j]
    return normalize(vector)

'''
Inputs :
    n : number of nodes
    prob_distrib : probability distribution of the positions
Output : Print the position on the x-axis and the probabilities of the walker being at each position of the y-axis
'''
def plot_distrib(n, prob_distrib):
    x = np.arange(n)
    y = []
    for i in range(n):
        if i in prob_distrib.keys():
            y.append(prob_distrib[i])
        else:
            y.append(float('nan'))
    plt.figure(figsize=(6,6), dpi=80)
    plt.xlabel("Position")
    plt.ylabel("Probabilities")
    plt.bar(x, y, width = 0.5, color='blue')

'''
Inputs :
    n : number of nodes
    coin_list : coin_list[i] contains the coin operator we want to apply on position i
Output : matrix U applying the coins operators on the coin space and the identity on the position space
'''
def get_u_position_dependent(n, coin_list):
    coin_operators = np.zeros((2*n, 2*n))
    for i in range(n):
        a = np.zeros((n, n))
        a[i][i] = 1
        coin_operators += np.kron(a,coin_list[i])
    return coin_operators

'''
EXAMPLE with no position-dependent coin operator

n = 8 # number of nodes (must be a power of 2)
n_step = 2 # number of steps

coin = (1/np.sqrt(2)) * np.array([[1,1], [1, -1]]) # Hadamard coin

pos_init = [1,0] # the walker starts on the node 0
coin_init = [1,0] # |0>

vector = init(n, pos_init, coin_init) # state vector of the walk

vector_after_walk = quantum_walk(vector, n, coin, n_step)

prob_distrib = decode(vector_after_walk,n)
plot_distrib(n, prob_distrib)
'''

'''
EXAMPLE with position-dependent coin operator

n = 8 # number of nodes (must be a power of 2)
n_step = 13 # number of steps

coin_list = []
Hadamard = np.array([[1,1],[1,-1]])/np.sqrt(2)
Z = np.array([[1,0],[0,-1]])
for i in range(n):
    if i % 2 == 0:
        coin_list.append(Hadamard)
    else:
        coin_list.append(Z)

pos_init = [1,0] # the walker starts on the node 0
coin_init = [1,0] # |0>

vector = init(n, pos_init, coin_init) # state vector of the walk

vector_after_walk = quantum_walk(vector, n, coin, n_step, position_dependent = True, coin_list = coin_list)

prob_distrib = decode(vector_after_walk,n)
'''
