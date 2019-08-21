#
# Structure of this code
#
# 1, import packages
# 2. set parameters and inputs that are necessary to create an Economy instance
# 3. save an instance as a pickle
# 4. run a main simulation code (this is a different code)
# 5. retrieve simulation result from a pickle
# 6. save simulation results
#
#


if __name__ == '__main__':
    
    #
    # import necessary packages
    #
    
    import numpy as np
    import time
    import subprocess
    from SCEconomy_hy_ns_lifecycle import Economy, split_shock
    from markov import calc_trans, Stationary
    import pickle
    from quantecon.markov.approximation import tauchen

    ###
    ### specify parameters and other inputs
    ###

    
    ### log file destination ###
    detailed_output_file = './log/test.txt'

    
    # prices
    p = 2.147770639542637 # relative price of S-goods
    rc = 0.06813837786011569 # interest rate
    

    # S-corp production function
    alpha    = 0.3 # capital share parameter for S-corp
    phi      = 0.15 # sweat capital share for S-corp
    # composite labor share parameter is defined by nu = 1. - alpha - phi


    # capital depreciation parameters
    delk     = 0.041 # physical capital depreciation rate
    delkap   = 0.041 # sweat ccapitaldepreciation rate for S-owners
    la       = 0.7 # 1-la sweat capital depreciation rate for C-workers

    # household preference
    beta     = 0.98 # discount rate
    eta      = 0.42 # utility weight on consumption
    mu       = 1.5  # risk aversion coeff. of utility

    # final good aggregator
    rho      = 0.01 # Elasticity of substitutions parameter between C-S goods
    ome      = 0.4786843155497944 #weight parameter for C-goods in CES final good aggregator.

    # linear tax
    tauc     = 0.065 # consumption tax
    taud     = 0.133 # dividend tax
    taup     = 0.36 # profit tax 

    # C-corp production function
    theta    = 0.5000702399881483 #capital share parameter for C-corporation Cobb-Douglas technology
    A        = 1.577707121233179 #TFP parameter for C-corporation Cobb-Douglas technology

    # parameters for sweat capital production function
    veps     = 0.418 # owner's time share
    vthet    = 1.0 - veps # C-good share
    zeta     = 1.0 # TFP term 

    # CES aggregator 
    upsilon  = 0.5 #elasticity parameter between owner's labor and employee's labor
    varpi    = 0.5553092396149117# share parameter on employee's labor.

    # other parameters
    chi      = 0.0 # borrowing constraint parameter a' >= chi ks    
    grate    = 0.02 # Growth rate of the economy


    #state space grids

    # a function that generates non equi-spaced grid
    def curvedspace(begin, end, curve, num=100):
        ans = np.linspace(0., (end - begin)**(1.0/curve), num) ** (curve) + begin
        ans[-1] = end #so that the last element is exactly end
        return ans

        
    agrid = curvedspace(0., 200., 2., 40) 
    kapgrid = curvedspace(0., 2.0, 2., 30)
    

    # productivity shock

    rho_z = 0.7
    sig_z = 0.1
    num_z = 5

    rho_eps = 0.7
    sig_eps = 0.1
    num_eps = 5
    

    mc_z   = tauchen(rho = rho_z  , sigma_u = sig_z  , m = 3, n = num_z) # discretize z
    mc_eps = tauchen(rho = rho_eps, sigma_u = sig_eps, m = 3, n = num_eps) # discretize z     

    # prob_z = mc_z.P
    # prob_eps = mc_eps.P
    # prob = np.kron(prob_eps, prob_z)

    prob = np.load('./DeBacker/prob_epsz.npy') # transition matrix from DeBacker et al.
    zgrid = np.exp(mc_z.state_values) ** 2.0
    epsgrid = np.exp(mc_eps.state_values) 
    
    is_to_iz = np.load('./input_data/is_to_iz.npy') #convert s to eps
    is_to_ieps = np.load('./input_data/is_to_ieps.npy') #convert s to z

    # lifecycle-specific parameters
    prob_yo = np.array([[44./45., 1./45.], [3./45., 42./45.]]) # transition matrix for young-old state
    #[[y -> y, y -> o], [o -> y, o ->o]]
    iota     = 1.0 # paternalistic discounting rate. 
    la_tilde = 0.1 # 1 - la_tilde is sweat capital depreciation rate
    tau_wo   = 0.5 # productivity eps is replaced by tau_wo*eps if the agent is old
    tau_bo   = 0.5 # productivity z   is replaced by tau_bo*z   if the agent is old
    trans_retire = 0.48 # receives this transfer if the agent is old.
    
    

    # GDP guess and GDP_indexed parameters (except for nonlinear tax)

    g_div_gdp = 0.133 # government expenditure relative to GDP
    yn_div_gdp = 0.266 # non-business production relative to GDP
    xnb_div_gdp = 0.110 # non-business consumption relative to GDP
    GDP_guess = 3.14 #a guess for GDP value. This needs to be consistent with simulated GDP

    
    g = g_div_gdp*GDP_guess # actual GDP 
    yn = yn_div_gdp*GDP_guess # actual non-business production
    xnb = xnb_div_gdp*GDP_guess # actual non-business consumption



    ###  nonlinear tax functions ###

    # business tax
    taub = np.array([.137, .185, .202, .238, .266, .280])
    bbracket = np.array([0.150, 0.319, 0.824, 2.085, 2.930]) # brackets relative to GDP
    scaling_b = GDP_guess #actual brackets are bracket * GDP
    
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
    sim_time = 500 # simulation length
    num_total_pop = 25_000 # population in simulatin

    num_suba_inner = 20 #the number of equi-spaced subgrid between agrid
    num_subkap_inner = 30 #the number of equi-spaced subgrid between kapgrid

    num_core = 640 # number of cores for parallel

    
    # computational parameters for exogenous shocks
    path_to_data_i_s = './tmp/data_i_s' # temporary directory for shock
    path_to_data_is_o = './tmp/data_is_o' # temporary directory for shock
    buffer_time = 2_000 # 


    ###
    ### end specify parameters and other inputs
    ###

    
    ### generate shocks and save them ###
    #save and split shocks for istate

    def generate_shock(prob, num_agent, num_time, buffer_time, save_dest, seed, init_state):

        np.random.seed(seed)
        data_rand = np.random.rand(num_agent, num_time+buffer_time)
        data_i = np.ones((num_agent, num_time+buffer_time), dtype = int)
        data_i[:, 0] = init_state
        calc_trans(data_i, data_rand, prob)
        data_i = data_i[:, buffer_time:]
        np.save(save_dest + '.npy' , data_i)
        split_shock(save_dest, num_agent, num_core)


    generate_shock(prob = prob,
                   num_agent = num_total_pop,
                   num_time = sim_time,
                   buffer_time = buffer_time,
                   save_dest = path_to_data_i_s,
                   seed = 0,
                   init_state = 7)

    generate_shock(prob = prob_yo,
                   num_agent = num_total_pop,
                   num_time = sim_time+1,
                   buffer_time = buffer_time,
                   save_dest = path_to_data_is_o,
                   seed = 2,
                   init_state = 0)

    
    # np.random.seed(0)
    # data_rand = np.random.rand(num_total_pop, sim_time+buffer_time)
    # data_i_s = np.ones((num_total_pop, sim_time+buffer_time), dtype = int)
    # data_i_s[:, 0] = 7 #initial state. it does not matter if simulation is long enough.
    # calc_trans(data_i_s, data_rand, prob)
    # data_i_s = data_i_s[:, buffer_time:]
    # np.save(path_to_data_i_s + '.npy' , data_i_s)
    # split_shock(path_to_data_i_s, num_total_pop, num_core)
    # del data_rand, data_i_s    

    # #save and split shocks for is_old
    # np.random.seed(2)
    # data_rand = np.random.rand(num_total_pop, sim_time+buffer_time+1) #+1 is added since this matters in calculation
    # data_is_o = np.ones((num_total_pop, sim_time+buffer_time+1), dtype = int)
    # data_is_o[:, 0] = 0 #initial state. it does not matter if simulation is long enough.
    # calc_trans(data_is_o, data_rand, prob_yo)
    # data_is_o = data_is_o[:, buffer_time:]
    # np.save(path_to_data_is_o + '.npy' , data_is_o)
    # split_shock(path_to_data_is_o, num_total_pop, num_core)
    # del data_rand, data_is_o
    # ### end generate shocks and save them ###    


    print('Solving the model with the given prices...')
    print('Do not simulate more than one models at the same time...')
    print('GDP_guess = ', GDP_guess)


    #create Economy
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

                   

    # set prices
    econ.set_prices(p = p, rc = rc)


    # save an Economy object and pass it to another program
    with open('econ.pickle', mode='wb') as f: pickle.dump(econ, f)

    # run the main code
    t0 = time.time()
    result = subprocess.run(['mpiexec', '-n', str(num_core), 'python', 'SCEconomy_hy_ns_lifecycle.py'], stdout=subprocess.PIPE)
    t1 = time.time()


    # write output in the log file
    f = open(detailed_output_file, 'ab') #use byte mode
    f.write(result.stdout)
    f.close()


    # receive a simulation result
    with open('econ.pickle', mode='rb') as f: econ = pickle.load(f)

        
    # print parameters
    econ.print_parameters()
    
    ## #calculate other important variables ###
    econ.calc_sweat_eq_value() 
    econ.calc_age()
    econ.simulate_other_vars()
    econ.calc_moments()    
    econ.save_result() # save simulation result under ./save_data/ by default


    
    
