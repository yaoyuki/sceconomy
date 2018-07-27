import numpy as np
import time
import subprocess
from SCEconomy_LSC_give_A import Economy, split_shock
from markov import calc_trans

import pickle


if __name__ == '__main__':

    def curvedspace(begin, end, curve, num=100):
        import numpy as np
        ans = np.linspace(0, (end - begin)**(1.0/curve), num) ** (curve) + begin
        ans[-1] = end #so that the last element is exactly end
        return ans


    ### LSC PT setup ###
    input_path = './input_data_LSC_pt/'

    agrid = np.load(input_path + 'agrid.npy')
    epsgrid = np.load(input_path + 'epsgrid.npy')

    
    is_to_iz = np.load(input_path + 'is_to_iz.npy')
    is_to_ieps = np.load(input_path + 'is_to_ieps.npy')


    # zgrid = (np.load(input_path + 'zgrid.npy') ** 2.0) * 0.4
    
    #insert a new zgrid data
    # zgrid_zt = np.load(input_path + 'zgrid_original.npy')
    zgrid_zt = np.load(input_path + 'zgrid_07_0025.npy') 
    zgrid_zp = np.exp([0.0, 0.2])
    zgrid = np.kron(zgrid_zp, zgrid_zt)
    zgrid = (zgrid**2.0) * 0.39


    # prob = np.load(input_path + 'prob.npy')
    # path_to_data_i_s = input_path + 'data_i_s'
   

    #insert new shock sequences
    prob_P = np.array([[1. - 0.5/0.5*(1.-0.99), 0.5/0.5*(1.-0.99)], [1. - 0.99, 0.99]])

    prob_T = np.load('./input_data_LSC_pt/prob_07_07_01_0025.npy')
    # prob_T = np.load('./input_data_LSC_pt/prob_T.npy')
    prob = np.kron(prob_P, prob_T)

    num_pop = 100_000
    sim_time_full = 2_000
    data_i_s = np.ones((num_pop, sim_time_full), dtype = int)
    data_rand = np.random.rand(num_pop, sim_time_full)
    calc_trans(data_i_s, data_rand, prob)

    np.save('./input_data_LSC_pt/data_i_s_tmp.npy', data_i_s[:,-1000:])
    split_shock('./input_data_LSC_pt/data_i_s_tmp', 100_000, 4)

    path_to_data_i_s = input_path + 'data_i_s_tmp'
    
    


    ###end LSC PT setup ###

    
    ###define additional parameters###
    num_core = 4 #7 or 8 must be the best for Anmol's PC. set 3 or 4 for Yuki's laptop

    # prices
    w_, p_, rc_ = 3.1157859589600867, 0.67360818042535, 0.06370317226696648
    
    ###end defining additional parameters###

    print('Solving the model with the given prices...')
    print('Do not simulate more than one models at the same time...')

    econ = Economy(agrid = agrid, epsgrid = epsgrid, zgrid = zgrid,
                   is_to_iz = is_to_iz, is_to_ieps = is_to_ieps, prob = prob,
                   path_to_data_i_s = path_to_data_i_s, alpha = 0.5)
    
    econ.set_prices(w = w_, p = p_, rc = rc_)
    with open('econ.pickle', mode='wb') as f: pickle.dump(econ, f)

    t0 = time.time()

    result = subprocess.run(['mpiexec', '-n', str(num_core), 'python', 'SCEconomy_LSC_give_A.py'], stdout=subprocess.PIPE)
    
    t1 = time.time()

    with open('econ.pickle', mode='rb') as f: econ = pickle.load(f)


    w = econ.w
    p = econ.p
    rc = econ.rc
    moms = econ.moms
    
    dist = np.sqrt(moms[0]**2.0 + moms[1]**2.0 + moms[2]**2.0)
    
    if w != w_ or  p != p_ or  rc != rc_:
        print('err: input prices and output prices do not coincide.')
        print('w = ', w, ', w_ = ', w_)
        print('p = ', p, ', p_ = ', p_)
        print('rc = ', rc, ', rc_ = ', rc_)

    
    
    econ.calc_moments()
    ###calculate other important variables###
    # econ.calc_sweat_eq_value()
    econ.calc_age()
    econ.simulate_other_vars()
    econ.save_result()
    

    
    
