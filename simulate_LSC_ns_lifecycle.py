import numpy as np
import time
import subprocess
from SCEconomy_LSC_ns_lifecycle import Economy, split_shock

import pickle


if __name__ == '__main__':

    import sys
    args = sys.argv    
    num_core = int(args[1])

    from markov import calc_trans, Stationary
    #generate shock sequene
    path_to_data_i_s = './tmp/data_i_s'
    path_to_data_is_o = './tmp/data_is_o'    
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
    split_shock(path_to_data_i_s, num_pop, num_core)
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
    split_shock(path_to_data_is_o, num_pop, num_core)
    del data_rand, data_is_o

    # taub = np.array([0.137, 0.185, 0.202, 0.238, 0.266, 0.28]) * 0.50 #large one
    # psib = np.array([0.007026139999999993, 0.02013013999999999, 0.03, 0.08398847999999996, 0.19024008000000006, 0.2648964800000001])
    # taup = 0.20
    
    

    

    def curvedspace(begin, end, curve, num=100):
        import numpy as np
        ans = np.linspace(0, (end - begin)**(1.0/curve), num) ** (curve) + begin
        ans[-1] = end #so that the last element is exactly end
        return ans


    # additional info
    
    agrid2 = curvedspace(0., 200., 2., 40)
    zgrid2 = np.load('./input_data/zgrid.npy') ** 2.0

    GDP_guess = 3.20

    taup = 0.36#*(1.0 - 0.278)
    taub = np.array([0.137, 0.185, 0.202, 0.238, 0.266, 0.28])# *(1.0 -  0.506) #large one
    # psib = np.array([-0.016705100000000014, 0.00993489999999998, 0.03, 0.13975679999999993, 0.35576280000000016, 0.5075368000000003])
    
    

    p_, rc_, ome_, theta_  = 1.4363987684178972, 0.06804621197252395, 0.4561128052733918, 0.5071181286751945,

 

    # 1.5209092097405632, 0.053192497033077685, 0.46388548260346824, 0.6002410397243539
    
    ###end defining additional parameters#

    print('Solving the model with the given prices...')
    print('Do not simulate more than one models at the same time...')


    econ = Economy(sim_time = 1000, num_total_pop  = 100_000,
        agrid = agrid2, zgrid = zgrid2, rho = 0.01, prob = prob, prob_yo = prob_yo,
                   ome = ome_,  theta = theta_,
                   path_to_data_i_s = path_to_data_i_s, path_to_data_is_o = path_to_data_is_o,
                   scaling_n = GDP_guess, scaling_b = GDP_guess, g = 0.133*GDP_guess, yn = 0.266*GDP_guess, xnb = 0.110*GDP_guess,
                   delk = 0.041, delkap = 0.041,  veps = 0.418, vthet = 1.0 - 0.418,
                   tauc = 0.065, taud = 0.133,
                   taup = taup, taub = taub# , psib = psib
                   #, epsgrid = epsgrid2
    )    

    
    econ.set_prices(p = p_, rc = rc_)
    
    with open('econ.pickle', mode='wb') as f: pickle.dump(econ, f)

    t0 = time.time()
    #don't forget to replace import argument
    result = subprocess.run(['mpiexec', '-n', str(num_core), 'python', 'SCEconomy_LSC_ns_lifecycle.py'], stdout=subprocess.PIPE)
    t1 = time.time()

    detailed_output_file = './log/test.txt'
    f = open(detailed_output_file, 'ab') #use byte mode
    f.write(result.stdout)
    f.close()


    with open('econ.pickle', mode='rb') as f: econ = pickle.load(f)


    p = econ.p
    rc = econ.rc
    ome = econ.ome
    theta = econ.theta
    
    if p != p_ or  rc != rc_ or ome != ome_  or theta != theta_ :
    #if p != p_ or  rc != rc_:
        print('err: input prices and output prices do not coincide.')
        print('p = ', p, ', p_ = ', p_)
        print('rc = ', rc, ', rc_ = ', rc_)
        print('ome = ', ome, ', ome_ = ', ome_)
        print('theta = ', theta, ', theta_ = ', theta_)        
    

    
    #calc main moments
    econ.print_parameters()
    
    ###calculate other important variables###
    ##econ.calc_sweat_eq_value()
    econ.calc_age()
    econ.simulate_other_vars()
    econ.calc_moments()    
    econ.save_result()
    
    

    
    
