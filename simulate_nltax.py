import numpy as np
import time
import subprocess
from SCEconomy_nltax import Economy, split_shock

import pickle


if __name__ == '__main__':

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

    
    import sys
    args = sys.argv    
    num_core = int(args[1])

    def curvedspace(begin, end, curve, num=100):
        import numpy as np
        ans = np.linspace(0, (end - begin)**(1.0/curve), num) ** (curve) + begin
        ans[-1] = end #so that the last element is exactly end
        return ans


    # additional info
    
    agrid2 = curvedspace(0., 200., 2., 40)
    kapgrid2 = curvedspace(0., 3., 2., 30)
    zgrid2 = np.load('./input_data/zgrid.npy') ** 2.0    
    
    

    p_, rc_, ome_, varpi_ = 1.417444459595774, 0.0610743812616079, 0.4627459750781605, 0.6056020599342775

    split_shock(path_to_shock, 100_000, num_core)
    
    ###end defining additional parameters#

    print('Solving the model with the given prices...')
    print('Do not simulate more than one models at the same time...')

    alpha = 0.3
    theta = 0.41

    ynb_p_gdp = 0.25
    xnb_p_gdp = 0.105
    g_p_gdp = 0.13
    
    pure_sweat_share = 0.10 #target
    s_emp_share = 0.30 #target

    yc_init = 0.8679
    # yc_init = 0.76

    # psib2 = np.array([0.12662323, 0.13995705, 0.15, 0.20493531, 0.3130503, 0.38901599])
    # taub2 = np.array([0.80*0.137, 0.80*0.185, 0.80*0.202, 0.89*0.238, 0.89 * 0.266, 0.89 * 0.28])

    GDP_implied = (1.-alpha + s_emp_share/(1. - s_emp_share)*(1.-theta))/((1.-alpha)*(1. - ynb_p_gdp) - pure_sweat_share)*yc_init

    econ = Economy(alpha = alpha, theta = theta, yn = ynb_p_gdp*GDP_implied, xnb = xnb_p_gdp*GDP_implied, g = g_p_gdp*GDP_implied,
                   scaling_n = GDP_implied, scaling_b = GDP_implied,
                   agrid = agrid2, kapgrid = kapgrid2, zgrid = zgrid2,  rho = 0.01, upsilon = 0.50, prob = prob, la = 0.7,
                   ome = ome_, varpi = varpi_, path_to_data_i_s = path_to_shock)
    #taub = taub2, psib = psib2, taup = 0.2,    
    
    econ.set_prices(p = p_, rc = rc_)
    with open('econ.pickle', mode='wb') as f: pickle.dump(econ, f)

    t0 = time.time()

    #don't forget to replace import argument
    result = subprocess.run(['mpiexec', '-n', str(num_core), 'python', 'SCEconomy_nltax.py'], stdout=subprocess.PIPE)
    
    t1 = time.time()

    detailed_output_file = './log/test.txt'
    f = open(detailed_output_file, 'ab') #use byte mode
    f.write(result.stdout)
    f.close()


    with open('econ.pickle', mode='rb') as f: econ = pickle.load(f)


    p = econ.p
    rc = econ.rc
    moms = econ.moms
    
    dist = np.sqrt(moms[0]**2.0 + moms[1]**2.0)
    
    if p != p_ or  rc != rc_:
        print('err: input prices and output prices do not coincide.')
        print('p = ', p, ', p_ = ', p_)
        print('rc = ', rc, ', rc_ = ', rc_)

    
    
    #calc main moments
    econ.print_parameters()
    econ.calc_moments()
    
    ###calculate other important variables###
    econ.calc_sweat_eq_value()
    econ.calc_age()
    econ.simulate_other_vars()
    econ.save_result()
    

    
    
