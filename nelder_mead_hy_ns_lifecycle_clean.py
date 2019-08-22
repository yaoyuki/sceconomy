import numpy as np
import time
import subprocess
import pickle
from quantecon.markov.approximation import tauchen    

from SCEconomy_hy_ns_lifecycle import Economy, split_shock
from markov import calc_trans, Stationary
from PiecewiseLinearTax import get_consistent_phi

from scipy.optimize import minimize

### log file destination ###

nd_log_file = '/cluster/shared/yaoxx366/log2/log.txt'
detailed_output_file = '/cluster/shared/yaoxx366/log2/detail.txt'


# initial prices and parameters

p_init = 2.147770639542637 # relative price of S-goods
rc_init = 0.06813837786011569 # interest rate

ome_init = 0.4786843155497944
varpi_init = 0.5553092396149117
theta_init = 0.5000702399881483

prices_init = [p_init, rc_init, ome_init, varpi_init, theta_init]



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
# ome      = 0.4786843155497944 #weight parameter for C-goods in CES final good aggregator.

# linear tax
tauc     = 0.065 # consumption tax
taud     = 0.133 # dividend tax
taup     = 0.36 # profit tax 

# C-corp production function
# theta    = 0.5000702399881483 #capital share parameter for C-corporation Cobb-Douglas technology
A        = 1.577707121233179 #TFP parameter for C-corporation Cobb-Douglas technology

# parameters for sweat capital production function
veps     = 0.418 # owner's time share
vthet    = 1.0 - veps # C-good share
zeta     = 1.0 # TFP term 

# CES aggregator 
upsilon  = 0.5 #elasticity parameter between owner's labor and employee's labor
# varpi    = 0.5553092396149117# share parameter on employee's labor.

# other parameters
chi      = 0.0 # borrowing constraint parameter a' >= chi ks    
grate    = 0.02 # Growth rate of the economy


# state space grids
# state space grid requires four parameters
# min, max, curvature, number of grid point
# if curvature is 1, the grid is equi-spaced.
# if curvature is larger than one, it puts more points near min


amin = 0.0 
amax = 200.
acurve = 2.0
num_a = 40

kapmin = 0.0
kapmax = 2.0
kapcurve = 2.0
num_kap = 30

# a function that generates non equi-spaced grid
def curvedspace(begin, end, curve, num=100):
    ans = np.linspace(0., (end - begin)**(1.0/curve), num) ** (curve) + begin
    ans[-1] = end #so that the last element is exactly end
    return ans
    
    
agrid = curvedspace(amin, amax, acurve, num_a) 
kapgrid = curvedspace(kapmin, kapmax, kapcurve, num_kap)

    
# productivity shock

rho_z = 0.7
sig_z = 0.1
num_z = 5

rho_eps = 0.7
sig_eps = 0.1
num_eps = 5


mc_z   = tauchen(rho = rho_z  , sigma_u = sig_z  , m = 3, n = num_z) # discretize z
mc_eps = tauchen(rho = rho_eps, sigma_u = sig_eps, m = 3, n = num_eps) # discretize eps

    
# prob_z = mc_z.P
# prob_eps = mc_eps.P
# prob = np.kron(prob_eps, prob_z)

prob_z   = np.loadtxt('./DeBacker/debacker_prob_z.npy') # read transition matrix from DeBacker
prob_eps = np.loadtxt('./DeBacker/debacker_prob_eps.npy') # read transition matrix from DeBacker
prob = np.kron(prob_eps, prob_z) 

# prob = np.load('./DeBacker/prob_epsz.npy') # transition matrix from DeBacker et al.
zgrid = np.exp(mc_z.state_values) ** 2.0
epsgrid = np.exp(mc_eps.state_values) 

is_to_iz = np.array([i for i in range(num_z) for j in range(num_eps)])
is_to_iz = np.array([j for i in range(num_z) for j in range(num_eps)])    
# is_to_iz = np.load('./input_data/is_to_iz.npy') #convert s to eps
# is_to_ieps = np.load('./input_data/is_to_ieps.npy') #convert s to z

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
bbracket_div_gdp = np.array([0.150, 0.319, 0.824, 2.085, 2.930]) # brackets relative to GDP
bbracket = bbracket_div_gdp * GDP_guess

# one intercept should be fixed
psib_fixed = 0.03 # value for the fixed intercept
bbracket_fixed = 2 # index for the fixed intercept
psib = get_consistent_phi(bbracket, taub, psib_fixed, bbracket_fixed) # obtain consistent intercepts

# labor income tax
taun = np.array([.2930, .3170, .3240, .3430, .3900, .4050, .4080, .4190])
nbracket_div_gdp = np.array([.1760, .2196, .2710, .4432, 0.6001, 1.4566, 2.7825]) # brackets relative to GDP
nbracket = nbracket_div_gdp * GDP_guess

# one intercept should be fixed
psin_fixed = 0.03 # value for the fixed intercept
nbracket_fixed = 5 # index for the fixed intercept
psin = get_consistent_phi(nbracket, taun, psin_fixed, nbracket_fixed) # obtain consistent intercepts



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


    #create an Economy instance
    econ = Economy(alpha = alpha,
                   beta = beta,
                   chi = chi,
                   delk = delk,
                   delkap = delkap,
                   eta = eta,
                   grate = grate,
                   la = la,
                   mu = mu,
                   ome = ome_, # target
                   phi = phi,
                   rho = rho,
                   tauc = tauc,
                   taud = taud,
                   taup = taup,
                   theta = theta, # target
                   veps = veps,
                   vthet = vthet,
                   zeta = zeta,
                   A = A,
                   upsilon = upsilon,
                   varpi = varpi, # varpi
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
                   psib = psib,
                   taun = taun,
                   nbracket = nbracket,
                   psin = psin,
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
    f.writelines(str(p_) + ', ' + str(rc_) + ', ' + str(ome_) + ', ' + str(varpi_)  + ', ' + str(theta_) +  ', ' + str(dist) + ', ' +  str(moms[0]) + ', ' + str(moms[1]) + ', ' + str(moms[2]) + ', ' + str(moms[4]) + ', ' + str(moms[5]) + ', ' + str(moms[7]) + ', ' + str(moms[8])  +  '\n')
  
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
    del data_rand

    #save and split shocks for is_old
    np.random.seed(2)
    data_rand = np.random.rand(num_total_pop, sim_time+buffer_time+1) #+1 is added since this matters in calculation
    data_is_o = np.ones((num_total_pop, sim_time+buffer_time+1), dtype = int)
    data_is_o[:, 0] = 0 #initial state. it does not matter if simulation is long enough.
    calc_trans(data_is_o, data_rand, prob_yo)
    data_is_o = data_is_o[:, buffer_time:]
    np.save(path_to_data_is_o + '.npy' , data_is_o)
    split_shock(path_to_data_is_o, num_total_pop, num_core)
    del data_rand
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

    
    
