import sys
args = sys.argv

import numpy as np
import time
import subprocess
### use modified version of SCEConomy module
from SCEconomy_nltax import Economy, split_shock

import pickle

p_init = float(args[1])
rc_init = float(args[2])
ome_init = float(args[3])
varpi_init = float(args[4])

num_core = args[5]

print('the code is running with ', num_core, 'cores...')
prices_init = [p_init, rc_init]
# prices_init = [p_init, rc_init, ome_init, varpi_init]


nd_log_file = './log/log.txt'
detailed_output_file = './log/detail.txt'


f = open(detailed_output_file, 'w')
f.close()

dist_min = 10000000.0
econ_save = None

def curvedspace(begin, end, curve, num=100):
    import numpy as np
    ans = np.linspace(0, (end - begin)**(1.0/curve), num) ** (curve) + begin
    ans[-1] = end #so that the last element is exactly end
    return ans

agrid2 = curvedspace(0., 200., 2., 40)
kapgrid2 = curvedspace(0., 3., 2., 30)
zgrid2 = np.load('./input_data/zgrid.npy') ** 2.
prob = np.load('./input_data/transition_matrix.npy')
path_to_data_i_s = './tmp/data_i_s'
psib2 = np.array([0.12662323, 0.13995705, 0.15, 0.20493531, 0.3130503, 0.38901599])
taub2 = np.array([0.80*0.137, 0.80*0.185, 0.80*0.202, 0.89*0.238, 0.89 * 0.266, 0.89 * 0.28])





def target(prices):
    global dist_min
    global econ_save
    
    p_ = prices[0]
    rc_ = prices[1]
    ome_ = ome_init
    varpi_ = varpi_init
    # ome_ = prices[2]
    # varpi_ = prices[3]
    
    
    # print('computing for the case w = {:f}, p = {:f}, rc = {:f}'.format(w_, p_, rc_), end = ', ')
    print('computing for the case p = {:f}, rc = {:f}'.format(p_, rc_), end = ', ')

    alpha = 0.4
    theta = 0.41
    # ynb = 0.451
    ynb_p_gdp = 0.25
    xnb_p_gdp = 0.105
    g_p_gdp = 0.13
    
    pure_sweat_share = 0.10 #target
    s_emp_share = 0.30 #target

    yc_init = 0.76 * 1.05 #1.0

    # GDP_implied = (1.-alpha + s_emp_share/(1. - s_emp_share)*(1.-theta)*yc_init + (1.-alpha)*ynb)/(1.-alpha - pure_sweat_share)
    
    GDP_implied = (1.-alpha + s_emp_share/(1. - s_emp_share)*(1.-theta))/((1.-alpha)*(1. - ynb_p_gdp) - pure_sweat_share)*yc_init

    econ = Economy(alpha = alpha, theta = theta, yn = ynb_p_gdp*GDP_implied, xnb = xnb_p_gdp*GDP_implied, g = g_p_gdp*GDP_implied,
                   scaling_n = GDP_implied, scaling_b = GDP_implied,
                   agrid = agrid2, kapgrid = kapgrid2, zgrid = zgrid2, prob = prob, rho = 0.01, upsilon = 0.50,
                   taub = taub2, taup = 0.2, psib = psib2,
                   ome = ome_, varpi = varpi_, path_to_data_i_s = path_to_data_i_s)

    econ.set_prices(p = p_, rc = rc_)
    
    with open('econ.pickle', mode='wb') as f: pickle.dump(econ, f)
    #with open('econ.pickle', mode='rb') as f: econ = pickle.load(f)
    t0 = time.time()

    #result = subprocess.run(['mpiexec', '-n', num_core, '--machinefile=node.hf' ,'python', 'SCEconomy_s_emp.py'], stdout=subprocess.PIPE)
    result = subprocess.run(['mpiexec', '-n', num_core ,'python', 'SCEconomy_nltax.py'], stdout=subprocess.PIPE)
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

    moms = econ.moms

            
        # mom0 = comm.bcast(mom0) #1. - Ecs/Eys
        # mom1 = comm.bcast(mom1) # 1. - (Ecc  + Ex+ (grate + delk)*(kc + Eks) + g + xnb - yn)/yc
        # mom2 = comm.bcast(mom2) # 1. - (tax_rev - tran - netb)/g
        # mom3 = comm.bcast(mom3) # 0.0
        # mom4 = comm.bcast(mom4) # Ens/En
        # mom5 = comm.bcast(mom5) # (p*Eys - (rs+delk)*Eks - w*Ens)/GDP
        # mom6 = comm.bcast(mom6) # nc
        # mom7 = comm.bcast(mom7) # 1. - EIc
        
    

    dist = np.sqrt(moms[0]**2.0 + moms[1]**2.0)
    # dist = np.sqrt(moms[0]**2.0 + moms[1]**2.0 + moms[2]**2.0 + 100.0*(moms[4] - 0.3)**2.0 +500.* (moms[5]-0.09)**2.0 + 100.*(moms[6] - 0.11)**2.0) #mom3 should be missing.
    # dist = np.sqrt(moms[0]**2.0 + moms[1]**2.0 + 100.0*(moms[4] - s_emp_share)**2.0 +500.* (moms[5]-pure_sweat_share)**2.0 + 100.*(moms[7] - 0.37)**2.0) #mom3 should be missing.
    # dist = np.sqrt(moms[0]**2.0 + moms[1]**2.0 + (moms[4]/s_emp_share - 1.)**2.0 +  (moms[5]/pure_sweat_share - 1.)**2.0) #mom3 should be missing.
    
    if p != p_ or  rc != rc_ or ome != ome_ or varpi != varpi_:
        print('err: input prices and output prices do not coincide.')
        print('p = ', p, ', p_ = ', p_)
        print('rc = ', rc, ', rc_ = ', rc_)
        print('ome = ', ome, ', ome_ = ', ome_)
        print('varpi = ', varpi, ', varpi_ = ', varpi_)


        
    print('dist = {:f}'.format(dist))

    f = open(nd_log_file, 'a')
    f.writelines(str(p) + ', ' + str(rc) + ', ' + str(ome) + ', ' + str(varpi) + ', ' + str(dist) + ', ' +  str(moms[0]) + ', ' + str(moms[1]) + ', ' + str(moms[2]) + ', ' + str(moms[4]) + ', ' + str(moms[5]) + ', ' + str(moms[7]) +  '\n')
    # f.writelines(str(p) + ', ' + str(rc) + ', ' + str(varpi) + ', ' + str(ome) + ', ' + str(theta) + ', ' +  str(dist) + ', ' +\
  
    f.close()
    
    if dist < dist_min:
        econ_save = econ
        dist_min = dist
    return dist

if __name__ == '__main__':

    f = open(nd_log_file, 'w')
    # f.writelines('w, p, rc, dist, mom0, mom1, mom2, mom3\n')
    f.writelines('p, rc, ome, varpi, dist, mom0, mom1, mom2, mom4, mom5, mom7\n')
    f.close()


    from markov import calc_trans, Stationary
    
    num_pop = 100_000
    sim_time = 3_000

    data_i_s = np.ones((num_pop, sim_time), dtype = int)
    #need to set initial state for zp
    data_i_s[:, 0] = 7
    prob = np.load('./input_data/transition_matrix.npy')    
    np.random.seed(0)
    data_rand = np.random.rand(num_pop, sim_time)
    calc_trans(data_i_s, data_rand, prob)
    data_i_s = data_i_s[:, 2000:]

    np.save(path_to_data_i_s + '.npy' , data_i_s)


    ### check
    f = open(nd_log_file, 'w')
    f.writelines(np.array_str(np.bincount(data_i_s[:,0]) / np.sum(np.bincount(data_i_s[:,0])), precision = 4, suppress_small = True) + '\n')
    f.writelines(np.array_str(Stationary(prob), precision = 4, suppress_small = True) + '\n')
    f.close()

    del data_i_s

    split_shock(path_to_data_i_s, 100_000, int(num_core))


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

    print('yc_init     = ', yc_init)    
    print('GDP Implied = ', GDP_implied)
    

    e.print_parameters()
    e.calc_moments()


    #
    #econ.calc_sweat_eq_value()
    #econ.simulate_other_vars()
    #econ.save_result()
    

    
    
