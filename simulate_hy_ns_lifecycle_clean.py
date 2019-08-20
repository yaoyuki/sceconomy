import numpy as np
import time
import subprocess
from SCEconomy_hy_ns_lifecycle import Economy, split_shock
from markov import calc_trans, Stationary

import pickle


if __name__ == '__main__':


    ### log file destination ###
    detailed_output_file = './log/test.txt'

    ###
    ### specify parameters and other inputs
    ###

    # prices

    p = 2.147770639542637
    rc = 0.06813837786011569
    
    # parameters
    alpha    = 0.3
    beta     = 0.98
    chi      = 0.0 
    delk     = 0.041
    delkap   = 0.041 
    eta      = 0.42
    grate    = 0.02 
    la       = 0.7 
    mu       = 1.5 
    ome      = 0.4786843155497944
    phi      = 0.15 
    rho      = 0.01
    tauc     = 0.065
    taud     = 0.133
    taup     = 0.36
    theta    = 0.5000702399881483
    veps     = 0.418
    vthet    = 1.0 - veps
    zeta     = 1.0
    A        = 1.577707121233179 #this should give yc = 1 (approx.) z^2 case
    upsilon  = 0.5
    varpi    = 0.5553092396149117



    #state space grids

    def curvedspace(begin, end, curve, num=100):
        ans = np.linspace(0., (end - begin)**(1.0/curve), num) ** (curve) + begin
        ans[-1] = end #so that the last element is exactly end
        return ans

    agrid = curvedspace(0., 200., 2., 40)
    kapgrid = curvedspace(0., 2.0, 2., 30)
    

    # productivity shock
    prob = np.load('./DeBacker/prob_epsz.npy') #state space (eps,z) is mapped into one dimension (s)
    zgrid = np.load('./input_data/zgrid.npy') ** 2.0
    epsgrid = np.load('./input_data/epsgrid.npy')
    is_to_iz = np.load('./input_data/is_to_iz.npy') #convert s to eps
    is_to_ieps = np.load('./input_data/is_to_ieps.npy') #convert s to z

    # age shocks
    prob_yo = np.array([[44./45., 1./45.], [3./45., 42./45.]]) #[[y -> y, y -> o], [o -> y, o ->o]]
    iota     = 1.0
    la_tilde = 0.1
    tau_wo   = 0.5 
    tau_bo   = 0.5
    trans_retire = 0.48 
    
    

    # GDP guess and GDP_indexed parameters (except for nonlinear tax)
    GDP_guess = 3.14
    g = 0.133*GDP_guess
    yn = 0.266*GDP_guess
    xnb = 0.110*GDP_guess



    ###  nonlinear tax functions ###

    # business tax
    taub = np.array([.137, .185, .202, .238, .266, .280])
    bbracket = np.array([0.150, 0.319, 0.824, 2.085, 2.930]) # brackets relative to GDP
    scaling_b = GDP_guess
    
    # one intercept should be fixed
    psib_fixed = 0.03 # value for the fixed intercept
    bbracket_fixed = 2 # index for the fixed intercept
    # to exeognously pin down intercepts,  directly determine psib
    # psib = None 


    # labor income tax
    taun = np.array([.2930, .3170, .3240, .3430, .3900, .4050, .4080, .4190])
    nbracket = np.array([.1760, .2196, .2710, .4432, 0.6001, 1.4566, 2.7825]) # brackets relative to GDP
    scaling_n = GDP_guess
    
    # one intercept should be fixed
    psin_fixed = 0.03 # value for the fixed intercept
    nbracket_fixed = 5 # index for the fixed intercept
    # to exeognously pin down intercepts,  directly determine psin
    # psin = None         

    

    # computational parameters
    sim_time = 500
    num_total_pop = 25_000

    num_suba_inner = 20
    num_subkap_inner = 30

    num_core = 640

    
    # computational parameters for exogenous shocks
    path_to_data_i_s = './tmp/data_i_s'
    path_to_data_is_o = './tmp/data_is_o'    
    buffer_time = 2_000


    ###
    ### end specify parameters and other inputs
    ###

    
    ### generate shocks and save them ###
    #save and split shocks for istate
    np.random.seed(0)
    data_rand = np.random.rand(num_total_pop, sim_time+buffer_time)
    data_i_s = np.ones((num_total_pop, sim_time+buffer_time), dtype = int)
    data_i_s[:, 0] = 7 #initial state. it does not matter if simulation is long enough.
    calc_trans(data_i_s, data_rand, prob)
    data_i_s = data_i_s[:, buffer_time:]
    np.save(path_to_data_i_s + '.npy' , data_i_s)
    split_shock(path_to_data_i_s, num_total_pop, num_core)
    del data_rand, data_i_s    

    #save and split shocks for is_old
    np.random.seed(2)
    data_rand = np.random.rand(num_total_pop, sim_time+buffer_time+1) #+1 is added since this matters in calculation
    data_is_o = np.ones((num_total_pop, sim_time+buffer_time+1), dtype = int)
    data_is_o[:, 0] = 0 #initial state. it does not matter if simulation is long enough.
    calc_trans(data_is_o, data_rand, prob_yo)
    data_is_o = data_is_o[:, buffer_time:]
    np.save(path_to_data_is_o + '.npy' , data_is_o)
    split_shock(path_to_data_is_o, num_total_pop, num_core)
    del data_rand, data_is_o
    ### end generate shocks and save them ###    


    print('Solving the model with the given prices...')
    print('Do not simulate more than one models at the same time...')
    print('GDP_guess = ', GDP_guess)


    econ = Economy(alpha = alpha,
                   beta = beta,
                   chi = chi,
                   delk = delk,
                   delkap = delkap,
                   eta = eta,
                   grate = grate,
                   la = la,
                   mu = mu,
                   ome = ome,
                   phi = phi,
                   rho = rho,
                   tauc = tauc,
                   taud = taud,
                   taup = taup,
                   theta = theta,
                   veps = veps,
                   vthet = vthet,
                   zeta = zeta,
                   A = A,
                   upsilon = upsilon,
                   varpi = varpi,
                   agrid = agrid,
                   kapgrid = kapgrid,
                   prob = prob,
                   zgrid = zgrid,
                   epsgrid = epsgrid,
                   is_to_iz = is_to_iz,
                   is_to_ieps = is_to_ieps,
                   prob_yo = prob_yo,
                   iota = iota,
                   la_tilde = la_tilde,
                   tau_wo = tau_wo,
                   tau_bo = tau_bo,
                   trans_retire = trans_retire,
                   g = g,
                   yn = yn,
                   xnb = xnb,
                   taub = taub,
                   bbracket = bbracket,
                   scaling_b = scaling_b,
                   psib_fixed = psib_fixed,
                   bbracket_fixed = bbracket_fixed,
                   taun = taun,
                   nbracket = nbracket,
                   scaling_n = scaling_n,
                   psin_fixed = psin_fixed,
                   nbracket_fixed = nbracket_fixed,
                   sim_time = sim_time,
                   num_total_pop = num_total_pop,
                   num_suba_inner = num_suba_inner,
                   num_subkap_inner = num_subkap_inner,
                   path_to_data_i_s = path_to_data_i_s,
                   path_to_data_is_o = path_to_data_is_o)

                   

    
    econ.set_prices(p = p, rc = rc)


    # save Econ object and pass it to another program
    with open('econ.pickle', mode='wb') as f: pickle.dump(econ, f)

    # run the main code
    t0 = time.time()
    result = subprocess.run(['mpiexec', '-n', str(num_core), 'python', 'SCEconomy_hy_ns_lifecycle.py'], stdout=subprocess.PIPE)
    t1 = time.time()


    # write output in the log file
    f = open(detailed_output_file, 'ab') #use byte mode
    f.write(result.stdout)
    f.close()


    # receive 
    with open('econ.pickle', mode='rb') as f: econ = pickle.load(f)

        
    #calc main moments
    econ.print_parameters()
    
    ###calculate other important variables###
    econ.calc_sweat_eq_value()
    econ.calc_age()
    econ.simulate_other_vars()
    econ.calc_moments()    
    econ.save_result()


    
    
