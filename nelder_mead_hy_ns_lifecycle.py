import sys
args = sys.argv

import numpy as np
import time
import subprocess
### use modified version of SCEConomy module
from SCEconomy_hy_ns_lifecycle import Economy, split_shock

import pickle

p_init = float(args[1])
rc_init = float(args[2])
ome_init = float(args[3])
varpi_init = float(args[4])
theta_init = float(args[5])

num_core = int(args[6])

print('the code is running with ', num_core, 'cores...')

# prices_init = [p_init, rc_init]
prices_init = [p_init, rc_init, ome_init, varpi_init, theta_init]
# prices_init = [p_init, rc_init,  theta_init]


# nd_log_file = './log/log.txt'
# detailed_output_file = './log/detail.txt'

nd_log_file = '/cluster/shared/yaoxx366/log2/log.txt'
detailed_output_file = '/cluster/shared/yaoxx366/log2/detail.txt'


f = open(detailed_output_file, 'w')
f.close()

dist_min = 10000000.0
econ_save = None


from markov import calc_trans, Stationary
#generate shock sequene
path_to_data_i_s = './tmp/data_i_s'
path_to_data_is_o = './tmp/data_is_o'    
num_pop = 25_000
sim_time = 2_500

prob = np.load('./DeBacker/prob_epsz.npy')
prob_yo = np.array([[44./45., 1./45.], [3./45., 42./45.]]) #[[y -> y, y -> o], [o -> y, o ->o]]


def curvedspace(begin, end, curve, num=100):
    import numpy as np
    ans = np.linspace(0, (end - begin)**(1.0/curve), num) ** (curve) + begin
    ans[-1] = end #so that the last element is exactly end
    return ans

agrid2 = curvedspace(0., 200., 2., 40)
kapgrid2 = curvedspace(0., 2., 2., 30)
zgrid2 = np.load('./input_data/zgrid.npy') ** 2.
# epsgrid2 = (np.load('./input_data/epsgrid.npy') ** 1.75) * 0.8
prob = np.load('./DeBacker/prob_epsz.npy') #DeBacker

pure_sweat_share = 0.090 # target
s_emp_share = 0.33 # target
xc_share = 0.134 # target
#w*nc/GDP = 0.22

GDP_guess = 3.70

# taup = 0.36*(1.0 - 0.278)
# taub = np.array([0.137, 0.185, 0.202, 0.238, 0.266, 0.28]) *(1.0 -  0.506) #large one
# psib = np.array([0.007026139999999993, 0.02013013999999999, 0.03, 0.08398847999999996, 0.19024008000000006, 0.2648964800000001])


# ome_ = ome_init
# varpi_ =  varpi_init
# theta_ =  theta_init

def target(prices):
    global dist_min
    global econ_save
    
    p_ = prices[0]
    rc_ = prices[1]
    ome_ = prices[2]
    varpi_ = prices[3]
    theta_ = prices[4]
    
    
    # print('computing for the case w = {:f}, p = {:f}, rc = {:f}'.format(w_, p_, rc_), end = ', ')
    print('computing for the case p = {:f}, rc = {:f}'.format(p_, rc_), end = ', ')
    
    ###set any additional condition/parameters
    ### alpha = 0.4 as default, and nu = 1. - phi - alpha
    #econ = Economy(agrid = agrid2, zgrid = zgrid2, path_to_data_i_s = path_to_data_i_s, rho = 0.01, ome = 0.6, varpi = 0.1)

    econ = Economy(agrid = agrid2, kapgrid = kapgrid2, zgrid = zgrid2, rho = 0.01, upsilon = 0.50, prob = prob,
                   ome = ome_, varpi = varpi_, theta = theta_,
                   path_to_data_i_s = path_to_data_i_s, path_to_data_is_o = path_to_data_is_o,
                   scaling_n = GDP_guess, scaling_b = GDP_guess, g = 0.133*GDP_guess, yn = 0.266*GDP_guess, xnb = 0.110*GDP_guess,
                   delk = 0.041, delkap = 0.041,  veps = 0.418, vthet = 1.0 - 0.418,
                   tauc = 0.065, taud = 0.133, taup = 0.36
                   #, epsgrid = epsgrid2
                   
                   
    )
                   


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
        
    p = econ.p
    rc = econ.rc
    ome = econ.ome
    varpi = econ.varpi
    theta = econ.theta
    
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
        
    

    # dist = np.sqrt(moms[0]**2.0 + moms[1]**2.0) #mom3 should be missing.
    # dist = np.sqrt(moms[0]**2.0 + moms[1]**2.0 + (moms[4]/s_emp_share - 1.)**2.0 +  (moms[5]/pure_sweat_share - 1.)**2.0) #mom3 should be missing.
    # dist = np.sqrt(moms[0]**2.0 + moms[1]**2.0 + (moms[4]/s_emp_share - 1.)**2.0 +  (moms[5]/pure_sweat_share - 1.)**2.0 + (moms[8]/xc_share - 1.)**2.0 ) #mom3 should be missing.
    dist = np.sqrt(5.*moms[0]**2.0 + 5.*moms[1]**2.0 + (moms[4]/s_emp_share - 1.)**2.0 +  (moms[5]/pure_sweat_share - 1.)**2.0 + (moms[8]/xc_share - 1.)**2.0 ) #mom3 should be missing.
   
    
    if p != p_ or  rc != rc_ or ome != ome_ or varpi != varpi_ or theta != theta_ :
    #if p != p_ or  rc != rc_:
        print('err: input prices and output prices do not coincide.')
        print('p = ', p, ', p_ = ', p_)
        print('rc = ', rc, ', rc_ = ', rc_)
        print('ome = ', ome, ', ome_ = ', ome_)
        print('varpi = ', varpi, ', varpi_ = ', varpi_)
        print('theta = ', theta, ', theta_ = ', theta_)        


        
    print('dist = {:f}'.format(dist))

    f = open(nd_log_file, 'a')
    f.writelines(str(p) + ', ' + str(rc) + ', ' + str(ome) + ', ' + str(varpi)  + ', ' + str(theta) +  ', ' + str(dist) + ', ' +  str(moms[0]) + ', ' + str(moms[1]) + ', ' + str(moms[2]) + ', ' + str(moms[4]) + ', ' + str(moms[5]) + ', ' + str(moms[7]) + ', ' + str(moms[8])  +  '\n')
    # f.writelines(str(p) + ', ' + str(rc) + ', ' + str(varpi) + ', ' + str(ome) + ', ' + str(theta) + ', ' +  str(dist) + ', ' +\
  
    f.close()
    
    if dist < dist_min:
        econ_save = econ
        dist_min = dist
    return dist

if __name__ == '__main__':

    f = open(nd_log_file, 'w')
    # f.writelines('w, p, rc, dist, mom0, mom1, mom2, mom3\n')
    f.writelines('p, rc, ome, varpi, dist, mom0, mom1, mom2, mom4, mom5, mom7, mom8\n')        
    f.close()


    ### generate shocks ###
    np.random.seed(0)
    data_rand = np.random.rand(num_pop, sim_time)
    data_i_s = np.ones((num_pop, sim_time), dtype = int)
    data_i_s[:, 0] = 7 #initial state. it does not matter if simulation is long enough.
    calc_trans(data_i_s, data_rand, prob)
    data_i_s = data_i_s[:, 2000:]
    np.save(path_to_data_i_s + '.npy' , data_i_s)
    split_shock(path_to_data_i_s, 25_000, num_core)
    del data_rand
    

    np.random.seed(2)
    data_rand = np.random.rand(num_pop, sim_time+1) #+1 is added since this matters in calculation
    data_is_o = np.ones((num_pop, sim_time+1), dtype = int)
    data_is_o[:, 0] = 0 #initial state. it does not matter if simulation is long enough.
    calc_trans(data_is_o, data_rand, prob_yo)
    data_is_o = data_is_o[:, 2000:]
    np.save(path_to_data_is_o + '.npy' , data_is_o)
    split_shock(path_to_data_is_o, 25_000, num_core)
    del data_rand

    ### end generate shocks ###
    

    ### check
    f = open(nd_log_file, 'w')
    f.writelines(np.array_str(np.bincount(data_i_s[:,0]) / np.sum(np.bincount(data_i_s[:,0])), precision = 4, suppress_small = True) + '\n')
    f.writelines(np.array_str(Stationary(prob), precision = 4, suppress_small = True) + '\n')
    
    f.writelines(np.array_str(np.bincount(data_is_o[:,0]) / np.sum(np.bincount(data_is_o[:,0])), precision = 4, suppress_small = True) + '\n')
    f.writelines(np.array_str(Stationary(prob_yo), precision = 4, suppress_small = True) + '\n')
    
    # f.writelines('yc_init = ' +  str(yc_init) + '\n')
    # f.writelines('GDP_implied = ' +  str(GDP_implied) + '\n')    
    f.close()

    del data_i_s, data_is_o


    nm_result = None
    from scipy.optimize import minimize

    tol_nm = 1.0e-4

    for i in range(5):
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
    
    # print('yc_init     = ', yc_init)    
    # print('GDP Implied = ', GDP_implied)


    e.print_parameters()
    e.calc_moments()


    #
    #econ.calc_sweat_eq_value()
    #econ.simulate_other_vars()
    #econ.save_result()
    

    
    
