import numpy as np
import time
import subprocess
from SCEconomy_LSC_nltax import Economy

import pickle


if __name__ == '__main__':

    def curvedspace(begin, end, curve, num=100):
        import numpy as np
        ans = np.linspace(0, (end - begin)**(1.0/curve), num) ** (curve) + begin
        ans[-1] = end #so that the last element is exactly end
        return ans


    agrid = curvedspace(0.0, 100., 2.0, 40)


    alpha = 0.3 #new!
    theta = 0.41
    ynb_p_gdp = 0.25
    xnb_p_gdp = 0.105
    g_p_gdp = 0.13
    
    pure_sweat_share = 0.10
    yc_init = 1.04
    
    GDP_implied = yc_init/(1. - ynb_p_gdp - pure_sweat_share/(1.-alpha))
    
    ynb = ynb_p_gdp*GDP_implied
    xnb = xnb_p_gdp*GDP_implied
    g = g_p_gdp*GDP_implied
    
    
    # taup = 0.20
    # taub = np.array([0.137, 0.185, 0.202, 0.238, 0.266, 0.28]) * 0.50 #large one
    # psib = np.array([0.12837754, 0.14071072, 0.15, 0.20081269, 0.30081419, 0.37107904])
    

    ### additional info
    zgrid2 = np.load('./input_data/zgrid.npy') ** 2.0

    # zgrid2 = np.load('./input_data/zgrid_09_0075.npy') ** 2.0
    # prob2 = np.load('./input_data/prob_epsz_07_09_01_0075.npy')

    path_to_shock = './tmp/data_i_s'
    from markov import calc_trans, Stationary
    
    num_pop = 100_000
    sim_time = 3_000

    data_i_s = np.ones((num_pop, sim_time), dtype = int)
    #need to set initial state for zp
    data_i_s[:, 0] = 7

    # prob = np.load('./input_data/transition_matrix.npy')
    prob = np.load('./DeBacker/prob_epsz.npy')
    np.random.seed(0)
    data_rand = np.random.rand(num_pop, sim_time)
    calc_trans(data_i_s, data_rand, prob)
    data_i_s = data_i_s[:, 2000:]

    np.save(path_to_shock + '.npy' , data_i_s)

    p_, rc_ , ome_ = 0.275384608013927, 0.0579181695442646 ,0.783161613400783
    # 0.2351573248046552, 0.05318390611686519, 0.7831616134007835


    taun = np.array([0.1465, 0.1585, 0.162,  0.1715, 0.195,  0.2025, 0.204,  0.2095])
    psin = np.array([0.08021118,  0.08744664,  0.09007978,  0.09889971,  0.13458096,  0.15, 0.15748521, 0.20991406])


    ###define additional parameters###
    num_core = 4 #7 or 8 must be the best for Anmol's PC. set 3 or 4 for Yuki's laptop


    ###end defining additional parameters###

    print('Solving the model with the given prices...')
    print('Do not simulate more than one models at the same time...')

    econ = Economy(path_to_data_i_s = path_to_shock, prob = prob, zgrid = zgrid2, agrid = agrid,
                   g = g, yn = ynb, xnb = xnb, ome = ome_,
                   scaling_n = GDP_implied, scaling_b = GDP_implied,
                   taun = taun, psin = psin,
                   alpha = alpha, theta = theta)
    #taub = taub, psib = psib,taup = taup,
    
    econ.set_prices(p = p_, rc = rc_)
    with open('econ.pickle', mode='wb') as f: pickle.dump(econ, f)

    t0 = time.time()

    result = subprocess.run(['mpiexec', '-n', str(num_core), 'python', 'SCEconomy_LSC_nltax.py'], stdout=subprocess.PIPE)
    
    t1 = time.time()

    with open('econ.pickle', mode='rb') as f: econ = pickle.load(f)


    # w = econ.w
    p = econ.p
    rc = econ.rc
    moms = econ.moms
    
    dist = np.sqrt(moms[0]**2.0 + moms[1]**2.0)
    
    if p != p_ or  rc != rc_:
        print('err: input prices and output prices do not coincide.')
        print('p = ', p, ', p_ = ', p_)
        print('rc = ', rc, ', rc_ = ', rc_)

        
    #c
    econ.print_parameters()
    
    econ.calc_moments()
    ###calculate other important variables###
    # econ.calc_sweat_eq_value()
    econ.calc_age()
    econ.simulate_other_vars()
    econ.save_result()
    

    
    
