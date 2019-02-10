import sys
sys.path.insert(0, './library/')


import numpy as np
import numba as nb
import time
from markov import Stationary

input_path = './save_data/'

def dropna(array):
    return array[np.isfinite(array)]

def get_transition(data_before, data_after, num_bins = 10, density = True, remove = None, full_output = False):


        ### input: data_before, data_after (1d-arrays, have the same length, can have nan if the data is missing)
        ###      : cannnot have inf or -inf

        ### output: transition matrix P[bin_before, bin_after] if density = True
        ###       : A matrix[bin_before, bin_after] = #[bin_before, bin_after] if density = False


        ### assertation ###

    assert len(data_before) == len(data_after), 'two data points should have the same length'
    assert np.nanmax(data_before) != np.inf, 'data_before has inf'
    assert np.nanmax(data_after) != np.inf, 'data_after has inf'
    assert np.nanmin(data_before) != -np.inf, 'data_before has -inf'
    assert np.nanmin(data_after) != -np.inf, 'data_after has -inf'
    
    ### end assertation ###
    
    #if num_bins = 10, includes 10,..., 90, 100. 0 is not included.
    deciles = np.arange(1., num_bins+1) * 10.
    
    data = np.ones((len(data_before), 2)) * -10000000.
    data[:,0] = data_before
    data[:,1] = data_after
    
    bins = np.nanpercentile(data.flatten(), deciles)
    bins[-1] = np.inf #the bins are defined as [x1, x2), so max of data will be eliminated without this mod.
    
    #     bins_m_inf = np.ones(len(bins)+1)
    #     bins_m_inf[0] = -np.inf
    #     bins_m_inf[1:] = bins[:]
    
    #     print(np.sum(np.isfinite(dropna(data))))
    #     print(np.histogram(dropna(data), bins_m_inf))
    
    del data
    
    
    #digitize numbers.
    before = np.digitize(data_before, bins)
    after = np.digitize(data_after, bins)
    #nan will be assinged to num_bins. these data will be eliminated
    cdn = (before != num_bins) & (after != num_bins)
    
    before = before[cdn]
    after = after[cdn]

    
    max_before = np.nanmax(before)
    max_after = np.nanmax(after)
    
    ans = np.zeros((max_before + 1, max_after + 1))
    
    for p in range(len(before)):
        
        i = before[p]
        j = after[p]
        
        if remove is not None:
            
            if i == remove or j == remove:
                pass
            else:
                ans[i,j] += 1
                
        else:
            ans[i,j] += 1
                
                
    if density:
        ans = (ans / np.sum(ans, axis = 1).reshape(max_before + 1,1))
                    
    if full_output:
        return ans, bins #this bins has only 1, 2,..., max - 1, inf. does not include min or max
    else:
        return ans


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
                                                                #             ans[i,j] = array[step*i:step*(i+1), step*j:step*(j+1)].sum()

    return ans


if __name__ == '__main__':

    ### set prices ###

    w =  3.1462116878265785
    p =  0.9667416127789383
    rc =  0.062123464982454815

    rbar =  0.04348642548771837
    rs =  0.04348642548771837

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
    

    
    
    
