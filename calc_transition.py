import sys
sys.path.insert(0, './library/')


import numpy as np
import numba as nb
import time
from markov import Stationary

input_path = './save_data/'

def dropna(array):
    return array[np.isfinite(array)]

#add get_transition

def shrink_mat(array):
    dim = 5
    step = 2

    sdist= Stationary(array)


    ans = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(dim):
            ans[i,j] = sdist[step*i]*array[step*i, step*j:step*j+2].sum() + \
                       sdist[step*i+1]*array[step*i+1, step*j:step*j+2].sum()

            ans[i,j] = ans[i,j] / sdist[step*i:step*i+2].sum()

    return ans


if __name__ == '__main__':

    ### set prices ###

    w =  3.176828946788351
    p =  1.0448855748548853
    rc =  0.060571736562196


    rbar =  0.0424002155935372
    rs =  0.0424002155935372
    

    delk = 0.05
    

    print('did you put correct prices?')


    ### import data ###
    data_i_s = np.load(input_path + 'data_i_s.npy')
    is_to_iz = np.load(input_path + 'is_to_iz.npy')
    zgrid = np.load(input_path + 'zgrid.npy')
    data_z = zgrid[is_to_iz[data_i_s]]
    is_to_ieps = np.load(input_path + 'is_to_ieps.npy')
    epsgrid = np.load(input_path + 'epsgrid.npy')
    data_eps = epsgrid[is_to_ieps[data_i_s]]
    data_n = np.load(input_path + 'data_n.npy')
    data_ns = np.load(input_path + 'data_ns.npy')
    data_ys = np.load(input_path + 'data_ys.npy')
    data_ks = np.load(input_path + 'data_ks.npy')
    data_x = np.load(input_path + 'data_x.npy')
    data_is_c = np.load(input_path + 'data_is_c.npy')

    ### import ###
    prob_z = np.loadtxt('./DeBacker/table4.csv', delimiter = ',')
    prob_z = (prob_z / prob_z.sum(axis = 1).reshape(10,1))
    prob_z = shrink_mat(prob_z)
    
    prob_eps = np.loadtxt('./DeBacker/table5.csv', delimiter = ',')
    prob_eps= (prob_eps / prob_eps.sum(axis = 1).reshape(10,1))
    prob_eps = shrink_mat(prob_eps)
    


    ### create income variable ###
    data_laborinc = w * data_eps * data_n
    data_bizinc = p*data_ys - (rs+delk)*data_ks - data_x - w*data_ns


    print('### labor income wepsn ###')

    print('')
    print('simulated result')

    prob_wepsn = get_transition(data_laborinc[:,-2], data_laborinc[:,-1], num_bins = 5, full_output = True)[0]
    print('transition of wepsn')
    print(np.array_str(prob_wepsn, precision = 4, suppress_small = True))
    print('implied SS of wespn')
    print(Stationary(prob_wepsn))
    print('')
    print('P_eps')
    print(prob_eps)
    print('SS of P_eps')    
    print(Stationary(prob_eps))
    

    print('')
    print('### biz income pys - (rs+delk)ks - wns - x ###')

    print('')
    print('simulated result')

    prob_bizinc = get_transition(data_bizinc[:,-2], data_bizinc[:,-1], num_bins = 5, full_output = True)[0]
    print('transition of bizinc')
    print(np.array_str(prob_bizinc, precision = 4, suppress_small = True))
    print('implied SS of bizinc')
    print(Stationary(prob_bizinc))
    print('')
    print('P_z')
    print(prob_z)
    print('SS P_z')
    print(Stationary(prob_z))
    

    
    
    
