import sys
args = sys.argv

import numpy as np
import time
import subprocess
from SCEconomy_hy_ns_lifecycle import Economy, split_shock
import pickle
from markov import calc_trans, Stationary
from scipy.optimize import minimize


### log file destination ###

nd_log_file = '/cluster/shared/yaoxx366/log2/log.txt'
detailed_output_file = '/cluster/shared/yaoxx366/log2/detail.txt'


###
### specify parameters and other inputs
###


# initial prices and parameters

p_init = 2.147770639542637
rc_init = 0.06813837786011569

ome_init = 0.4786843155497944
varpi_init = 0.5553092396149117
theta_init = 0.5000702399881483

prices_init = [p_init, rc_init, ome_init, varpi_init, theta_init]
    
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
# ome      = 0.4786843155497944
phi      = 0.15 
rho      = 0.01
tauc     = 0.065
taud     = 0.133
taup     = 0.36
# theta    = 0.5000702399881483
veps     = 0.418
vthet    = 1.0 - veps
zeta     = 1.0
A        = 1.577707121233179 #this should give yc = 1 (approx.) z^2 case
upsilon  = 0.5
# varpi    = 0.5553092396149117



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
GDP_guess = 3.34
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
sim_time = 1000
num_total_pop = 100_000

num_suba_inner = 20
num_subkap_inner = 30

num_core = 640


# computational parameters for exogenous shocks
path_to_data_i_s = './tmp/data_i_s'
path_to_data_is_o = './tmp/data_is_o'    
buffer_time = 2_000


### calibration target

pure_sweat_share = 0.090 # target
s_emp_share = 0.33 # target
xc_share = 0.134 # target
#w*nc/GDP = 0.22


### nelder mead option
tol_nm = 1.0e-4 #if the NM returns a value large than it, the NM restarts

###
### end specify parameters and other inputs
###



dist_min = 10000000.0
econ_save = None

def target(prices):
    global dist_min
    global econ_save
    
    p_ = prices[0]
    rc_ = prices[1]
    ome_ = prices[2]
    varpi_ = prices[3]
    theta_ = prices[4]
    
    

    print('computing for the case p = {:f}, rc = {:f}'.format(p_, rc_), end = ', ')

    econ = Economy(alpha = alpha,
                   beta = beta,
                   chi = chi,
                   delk = delk,
                   delkap = delkap,
                   eta = eta,
                   grate = grate,
                   la = la,
                   mu = mu,
                   ome = ome_, #target
                   phi = phi,
                   rho = rho,
                   tauc = tauc,
                   taud = taud,
                   taup = taup,
                   theta = theta_ , #target
                   veps = veps,
                   vthet = vthet,
                   zeta = zeta,
                   A = A,
                   upsilon = upsilon,
                   varpi = varpi_, #target
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
    

    econ.set_prices(p = p_, rc = rc_)
    
    with open('econ.pickle', mode='wb') as f: pickle.dump(econ, f)
    #with open('econ.pickle', mode='rb') as f: econ = pickle.load(f)
    t0 = time.time()
    result = subprocess.run(['mpiexec', '-n', str(num_core) ,'python', 'SCEconomy_hy_ns_lifecycle.py'], stdout=subprocess.PIPE)
    t1 = time.time()
    

    f = open(detailed_output_file, 'ab') #use byte mode
    f.write(result.stdout)
    f.close()
    
    print('etime: {:f}'.format(t1 - t0), end = ', ')

    time.sleep(1)

    with open('econ.pickle', mode='rb') as f: econ = pickle.load(f)
    
    moms = econ.moms

            
    # mom0 = comm.bcast(mom0) #1. - Ecs/Eys
    # mom1 = comm.bcast(mom1) # 1. - (Ecc  + Ex+ (grate + delk)*(kc + Eks) + g + xnb - yn)/yc
    # mom2 = comm.bcast(mom2) # 1. - (tax_rev - tran - netb)/g
    # mom3 = comm.bcast(mom3) # 0.0
    # mom4 = comm.bcast(mom4) # Ens/En
    # mom5 = comm.bcast(mom5) # (p*Eys - (rs+delk)*Eks - w*Ens)/GDP
    # mom6 = comm.bcast(mom6) # nc
    # mom7 = comm.bcast(mom7) # 1. - EIc
    # mom8 = comm.bcast(mom8) # xc/GDP        
    

    # dist = np.sqrt(moms[0]**2.0 + moms[1]**2.0) #if targets are just market clearing
    dist = np.sqrt(5.*moms[0]**2.0 + 5.*moms[1]**2.0 +\
                   (moms[4]/s_emp_share - 1.)**2.0 +\
                   (moms[5]/pure_sweat_share - 1.)**2.0 +\
                   (moms[8]/xc_share - 1.)**2.0 ) 
   
    
        
    print('dist = {:f}'.format(dist))

    f = open(nd_log_file, 'a')
    f.writelines(str(p) + ', ' + str(rc) + ', ' + str(ome) + ', ' + str(varpi)  + ', ' + str(theta) +  ', ' + str(dist) + ', ' +  str(moms[0]) + ', ' + str(moms[1]) + ', ' + str(moms[2]) + ', ' + str(moms[4]) + ', ' + str(moms[5]) + ', ' + str(moms[7]) + ', ' + str(moms[8])  +  '\n')
  
    f.close()
    
    if dist < dist_min:
        econ_save = econ
        dist_min = dist
    return dist

if __name__ == '__main__':

    f = open(nd_log_file, 'w')
    f.writelines('p, rc, ome, varpi,theta, dist, mom0, mom1, mom2, mom4, mom5, mom7, mom8\n')        
    f.close()



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
    

    ### check
    f = open(nd_log_file, 'a')
    f.writelines(np.array_str(np.bincount(data_i_s[:,0]) / np.sum(np.bincount(data_i_s[:,0])), precision = 4, suppress_small = True) + '\n')
    f.writelines(np.array_str(Stationary(prob), precision = 4, suppress_small = True) + '\n')
    
    f.writelines(np.array_str(np.bincount(data_is_o[:,0]) / np.sum(np.bincount(data_is_o[:,0])), precision = 4, suppress_small = True) + '\n')
    f.writelines(np.array_str(Stationary(prob_yo), precision = 4, suppress_small = True) + '\n')
    
    f.close()

    del data_i_s, data_is_o


    nm_result = None

    for i in range(5): # repeat up to 5 times
        nm_result = minimize(target, prices_init, method='Nelder-Mead', tol = tol_nm)

        if nm_result.fun < tol_nm: #1.0e-3
            break
        else:
            prices_init = nm_result.x #restart

    f = open(nd_log_file, 'a')
    f.write(str(nm_result))
    f.close()
    

    
    ###calculate other important variables###
    econ = econ_save
    with open('econ.pickle', mode='wb') as f: pickle.dump(econ, f)

    e = econ

    print('')

    print('agrid')
    print(e.agrid)

    print('kapgrid')
    print(e.kapgrid)

    print('zgrid')
    print(e.zgrid)

    print('epsgrid')
    print(e.epsgrid)

    print('prob')
    print(e.prob)

    print('prob_yo')
    print(e.prob_yo)
    print('GDP_guess = ', GDP_guess)    
    print('')    
    

    e.print_parameters()
    e.calc_moments()

    
    
