import numpy as np
import time
import subprocess
from SCEconomy_hy_ns_lifecycle_sales import Economy, split_shock

import pickle


if __name__ == '__main__':

    import sys
    args = sys.argv    
    num_core = int(args[1])

    from markov import calc_trans, Stationary
    #generate shock sequene
    path_to_data_i_s = './tmp/data_i_s'
    path_to_data_is_o = './tmp/data_is_o'
    path_to_data_sales_shock = './tmp/data_sales_shock'        
    num_pop = 100_000
    sim_time = 3_000

    #save and split shocks for istate
    # prob = np.load('./input_data/transition_matrix.npy')
    prob = np.load('./DeBacker/prob_epsz.npy')
    np.random.seed(0)
    data_rand = np.random.rand(num_pop, sim_time)
    data_i_s = np.ones((num_pop, sim_time), dtype = int)
    data_i_s[:, 0] = 7 #initial state. it does not matter if simulation is long enough.
    calc_trans(data_i_s, data_rand, prob)
    data_i_s = data_i_s[:, 2000:]
    np.save(path_to_data_i_s + '.npy' , data_i_s)
    split_shock(path_to_data_i_s, 100_000, num_core)
    del data_rand, data_i_s    

    #save and split shocks for is_old
    prob_yo = np.array([[44./45., 1./45.], [3./45., 42./45.]]) #[[y -> y, y -> o], [o -> y, o ->o]]
    np.random.seed(2)
    data_rand = np.random.rand(num_pop, sim_time+1) #+1 is added since this matters in calculation
    data_is_o = np.ones((num_pop, sim_time+1), dtype = int)
    data_is_o[:, 0] = 0 #initial state. it does not matter if simulation is long enough.
    calc_trans(data_is_o, data_rand, prob_yo)
    data_is_o = data_is_o[:, 2000:]
    np.save(path_to_data_is_o + '.npy' , data_is_o)
    split_shock(path_to_data_is_o, 100_000, num_core)
    del data_rand, data_is_o


    #prob  of being 1 is 0.15
    # data_sales_shock = np.random.choice(2, (num_pop, sim_time), p = [0.50, 0.50])
    data_sales_shock = np.random.choice(2, (num_pop, sim_time), p = [0.85, 0.15])
    data_sales_shock = data_sales_shock[:, 2000:]
    np.save(path_to_data_sales_shock + '.npy' , data_sales_shock)
    split_shock(path_to_data_sales_shock, 100_000, num_core)
    del data_sales_shock


    # #save and split shocks for is_old
    # prob_sales = np.array([[0.5, 0.5], [0.5, 0.5]]) #[[y -> y, y -> o], [o -> y, o ->o]]
    # np.random.seed(3)
    # data_rand = np.random.rand(num_pop, sim_time) #+1 is added since this matters in calculation
    # data_sales_shock = np.ones((num_pop, sim_time), dtype = int)
    # data_sales_shock[:, 0] = 0 #initial state. it does not matter if simulation is long enough.
    # calc_trans(data_sales_shock, data_rand, prob_sales)
    # data_sales_shock = data_sales_shock[:, 2000:]
    # np.save(path_to_data_sales_shock + '.npy' , data_sales_shock)
    # split_shock(path_to_data_sales_shock, 100_000, num_core)
    # del data_rand, data_sales_shock
    

    # taub = np.array([0.137, 0.185, 0.202, 0.238, 0.266, 0.28]) * 0.50 #large one
    # psib = np.array([0.007026139999999993, 0.02013013999999999, 0.03, 0.08398847999999996, 0.19024008000000006, 0.2648964800000001])
    taup = 0.20
    
    

    

    def curvedspace(begin, end, curve, num=100):
        import numpy as np
        ans = np.linspace(0, (end - begin)**(1.0/curve), num) ** (curve) + begin
        ans[-1] = end #so that the last element is exactly end
        return ans


    # additional info
    
    agrid2 = curvedspace(0., 200., 2., 40)
    kapgrid2 = curvedspace(0., 3.0, 2., 30)
    zgrid2 = np.load('./input_data/zgrid.npy') ** 2.0    
    

    p_, rc_, pkap_, kapbar_, ome_, varpi_ = 1.6959895478680314, 0.0022859333983255795, 0.012559094599466355, 0.05827055544624726, 0.46388548260346824, 0.6002410397243539
    # 1.6120266141599342, 0.004489583936630955, 0.00618750541475882, 0.07182001341947838, 0.46388548260346824, 0.6002410397243539
    
    ###end defining additional parameters#

    print('Solving the model with the given prices...')
    print('Do not simulate more than one models at the same time...')


    econ = Economy(agrid = agrid2, kapgrid = kapgrid2, zgrid = zgrid2, rho = 0.01, upsilon = 0.50, prob = prob,
                   ome = ome_, varpi = varpi_, path_to_data_i_s = path_to_data_i_s, path_to_data_is_o = path_to_data_is_o,
                   scaling_n = 1.82, scaling_b = 1.82, taup = taup
    )
    
    econ.set_prices(p = p_, rc = rc_, pkap = pkap_, kapbar = kapbar_)
    
    with open('econ.pickle', mode='wb') as f: pickle.dump(econ, f)

    t0 = time.time()
    #don't forget to replace import argument
    result = subprocess.run(['mpiexec', '-n', str(num_core), 'python', 'SCEconomy_hy_ns_lifecycle_sales.py'], stdout=subprocess.PIPE)
    t1 = time.time()

    detailed_output_file = './log/test.txt'
    f = open(detailed_output_file, 'ab') #use byte mode
    f.write(result.stdout)
    f.close()


    with open('econ.pickle', mode='rb') as f: econ = pickle.load(f)


    p = econ.p
    rc = econ.rc
    pkap = econ.pkap
    kapbar = econ.kapbar
    
    # moms = econ.moms
    
    # dist = np.sqrt(moms[0]**2.0 + moms[1]**2.0)
    
    # if p != p_ or  rc != rc_:
    #     print('err: input prices and output prices do not coincide.')
    #     print('p = ', p, ', p_ = ', p_)
    #     print('rc = ', rc, ', rc_ = ', rc_)

    
    #calc main moments
    econ.print_parameters()
    
    ###calculate other important variables###
    ##econ.calc_sweat_eq_value()
    econ.calc_age()
    econ.simulate_other_vars()
    econ.calc_moments()    
    econ.save_result()


    
    
