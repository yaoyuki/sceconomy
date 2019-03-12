import sys
args = sys.argv

import numpy as np
import time
import subprocess
### use modified version of SCEConomy module
from SCEconomy_LSC_nltax import Economy, split_shock

import pickle

p_init = float(args[1])
rc_init = float(args[2])
ome_init = float(args[3])
num_core = args[4]

print('the code is running with ', num_core, 'cores...')
prices_init = [p_init, rc_init, ome_init]


nd_log_file = './log/log.txt'
detailed_output_file = './log/detailed.txt'

f = open(detailed_output_file, 'w')
f.close()

dist_min = 10000000.0
econ_save = None
zgrid2 = np.load('./input_data/zgrid.npy') ** 2.0
prob = np.load('./DeBacker/prob_epsz.npy') #DeBacker
path_to_data_i_s = './tmp/data_i_s'

def curvedspace(begin, end, curve, num=100):
    import numpy as np
    ans = np.linspace(0, (end - begin)**(1.0/curve), num) ** (curve) + begin
    ans[-1] = end #so that the last element is exactly end
    return ans


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
    



def target(prices):
    global dist_min
    global econ_save

    p_ = prices[0]
    rc_ = prices[1]
    ome_ = prices[2]

    print('computing for the case p = {:f}, rc = {:f}'.format(p_, rc_), end = ', ')
    
    ###set any additional condition/parameters


    econ = Economy(path_to_data_i_s = path_to_data_i_s, prob = prob, zgrid = zgrid2,
                   g = g, yn = ynb, xnb = xnb, ome = ome_,
                   scaling_n = GDP_implied, scaling_b = GDP_implied,
                   alpha = alpha, theta = theta)


    econ.set_prices(p = p_, rc = rc_)
    
    with open('econ.pickle', mode='wb') as f: pickle.dump(econ, f)
    #with open('econ.pickle', mode='rb') as f: econ = pickle.load(f)
    t0 = time.time()

    result = subprocess.run(['mpiexec', '-n', num_core, 'python', 'SCEconomy_LSC_nltax.py'], stdout=subprocess.PIPE)
    t1 = time.time()
    

    f = open(detailed_output_file, 'ab') #use byte mode
    f.write(result.stdout)
    f.close()
    
    print('etime: {:f}'.format(t1 - t0), end = ', ')

    time.sleep(1)

    with open('econ.pickle', mode='rb') as f: econ = pickle.load(f)

    w = econ.w
    p = econ.p
    rc = econ.rc
    moms = econ.moms
    
    dist = np.sqrt(moms[0]**2.0 + moms[1]**2.0 + (1. - moms[3]/pure_sweat_share )**2.0)
    
    if p != p_ or  rc != rc_:
        print('err: input prices and output prices do not coincide.')
        print('p = ', p, ', p_ = ', p_)
        print('rc = ', rc, ', rc_ = ', rc_)
       # return
    
    print('dist = {:f}'.format(dist))

    f = open(nd_log_file, 'a')
    f.writelines(str(p) + ', ' + str(rc) + ', ' + str(ome_) + ', ' + str(dist) + ', ' +  str(moms[0]) + ', ' + str(moms[1]) + ', ' + str(moms[2]) + ', ' + str(moms[3]) + '\n')
    # f.writelines(str(p) + ', ' + str(trans) + ', ' + str(ome_) + ', ' + str(dist) + ', ' +  str(moms[0]) + ', ' + str(moms[1]) + ', ' + str(moms[2]) + ', ' + str(moms[3]) + '\n')
    f.close()
    
    if dist < dist_min:
        econ_save = econ
        dist_min = dist
    return dist

if __name__ == '__main__':

    f = open(nd_log_file, 'w')
    f.writelines('p, rc, ome, dist, mom0, mom1, mom2, mom3\n')
    f.close()

    #load shocks
    from markov import calc_trans, Stationary
    
    num_pop = 100_000
    sim_time = 3_000

    data_i_s = np.ones((num_pop, sim_time), dtype = int)
    #need to set initial state for zp
    data_i_s[:, 0] = 7


    np.random.seed(0)
    data_rand = np.random.rand(num_pop, sim_time)
    calc_trans(data_i_s, data_rand, prob)
    data_i_s = data_i_s[:, 2000:]

    np.save(path_to_data_i_s + '.npy' , data_i_s)

    ### check
    f = open(nd_log_file, 'a')
    f.writelines(np.array_str(np.bincount(data_i_s[:,0]) / np.sum(np.bincount(data_i_s[:,0])), precision = 4, suppress_small = True) + '\n')
    f.writelines(np.array_str(Stationary(prob), precision = 4, suppress_small = True) + '\n')
    # f.writelines('yc_init = ' +  str(yc_init) + '\n')
    # f.writelines('GDP_implied = ' +  str(GDP_implied) + '\n')    
    f.close()

    del data_i_s
    
    split_shock(path_to_data_i_s, 100_000, int(num_core))


    nm_result = None
    from scipy.optimize import minimize

    for i in range(5):
        nm_result = minimize(target, prices_init, method='Nelder-Mead')

        if nm_result.fun < 1.0e-3:
            break
        else:
            prices_init = nm_result.x #restart

    f = open(nd_log_file, 'a')
    f.write(str(nm_result))
    f.close()


    
    ###calculate other important variables###
    econ = econ_save
    with open('econ.pickle', mode='wb') as f: pickle.dump(econ, f)

    #
    #econ.calc_sweat_eq_value()
    #econ.simulate_other_vars()
    #econ.save_result()
    

    
    
