import sys
args = sys.argv

import numpy as np
import time
import subprocess
### use modified version of SCEConomy module

from SCEconomy_LSC_give_A import Economy, split_shock
from markov import calc_trans

import pickle

w_init = float(args[1])
p_init = float(args[2])
rc_init = float(args[3])
num_core = args[4]

print('the code is running with ', num_core, 'cores...')
prices_init = [w_init, p_init, rc_init]


nd_log_file = '/home/ec2-user/Dropbox/case1/log_alpha_03_chi_1_taum_02.txt'
detailed_output_file = '/home/ec2-user/Dropbox/case1/detailed_output_alpha_03_chi_1_taum_02.txt'

f = open(detailed_output_file, 'w')
f.close()

dist_min = 10000000.0
econ_save = None

def curvedspace(begin, end, curve, num=100):
    import numpy as np
    ans = np.linspace(0, (end - begin)**(1.0/curve), num) ** (curve) + begin
    ans[-1] = end #so that the last element is exactly end
    return ans



### LSC PT setup ###
input_path = './input_data_LSC_pt/'

agrid = np.load(input_path + 'agrid.npy')
epsgrid = np.load(input_path + 'epsgrid.npy')

    
is_to_iz = np.load(input_path + 'is_to_iz.npy')
is_to_ieps = np.load(input_path + 'is_to_ieps.npy')


# zgrid = (np.load(input_path + 'zgrid.npy') ** 2.0) * 0.4
    
#insert a new zgrid data
# zgrid_zt = np.load(input_path + 'zgrid_original.npy')
zgrid_zt = np.load(input_path + 'zgrid_07_0025.npy') 
zgrid_zp = np.exp([0.0, 0.2])
zgrid = np.kron(zgrid_zp, zgrid_zt)
zgrid = (zgrid**2.0) * 0.39


# prob = np.load(input_path + 'prob.npy')
# path_to_data_i_s = input_path + 'data_i_s'


#insert new shock sequences
prob_P = np.array([[1. - 0.5/0.5*(1.-0.99), 0.5/0.5*(1.-0.99)], [1. - 0.99, 0.99]])

prob_T = np.load('./input_data_LSC_pt/prob_07_07_01_0025.npy')
# prob_T = np.load('./input_data_LSC_pt/prob_T.npy')
prob = np.kron(prob_P, prob_T)

num_pop = 100_000
sim_time_full = 2_000
data_i_s = np.ones((num_pop, sim_time_full), dtype = int)
data_rand = np.random.rand(num_pop, sim_time_full)
calc_trans(data_i_s, data_rand, prob)

np.save('./input_data_LSC_pt/data_i_s_tmp.npy', data_i_s[:,-1000:])
split_shock('./input_data_LSC_pt/data_i_s_tmp', 100_000, int(num_core))

path_to_data_i_s = input_path + 'data_i_s_tmp'

###end LSC PT setup ###




def target(prices):
    global dist_min
    global econ_save
    
    w_ = prices[0]
    p_ = prices[1]
    rc_ = prices[2]
    
    print('computing for the case w = {:f}, p = {:f}, rc = {:f}'.format(w_, p_, rc_), end = ', ')
    
    ###set any additional condition/parameters
    econ = Economy(agrid = agrid, epsgrid = epsgrid, zgrid = zgrid,
                   is_to_iz = is_to_iz, is_to_ieps = is_to_ieps, prob = prob,
                   path_to_data_i_s = path_to_data_i_s, alpha = 0.3, taum = 0.20, chi = 1.0)
    

    econ.set_prices(w = w_, p = p_, rc = rc_)
    
    with open('econ.pickle', mode='wb') as f: pickle.dump(econ, f)
    #with open('econ.pickle', mode='rb') as f: econ = pickle.load(f)
    t0 = time.time()

    result = subprocess.run(['mpiexec', '-n', num_core, 'python', 'SCEconomy_LSC_give_A.py'], stdout=subprocess.PIPE)
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
    
    dist = np.sqrt(moms[0]**2.0 + moms[1]**2.0 + moms[2]**2.0)
    
    if w != w_ or  p != p_ or  rc != rc_:
        print('err: input prices and output prices do not coincide.')
        print('w = ', w, ', w_ = ', w_)
        print('p = ', p, ', p_ = ', p_)
        print('rc = ', rc, ', rc_ = ', rc_)
       # return
    
    print('dist = {:f}'.format(dist))

    f = open(nd_log_file, 'a')
    f.writelines(str(w) + ', ' + str(p) + ', ' + str(rc) + ', ' + str(dist) + ', ' +  str(moms[0]) + ', ' + str(moms[1]) + ', ' + str(moms[2]) + ', ' + str(moms[3]) + '\n')
    f.close()
    
    if dist < dist_min:
        econ_save = econ
        dist_min = dist
    return dist

if __name__ == '__main__':

    f = open(nd_log_file, 'w')
    f.writelines('w, p, rc, dist, mom0, mom1, mom2, mom3\n')
    f.close()

    nm_result = None
    from scipy.optimize import minimize

    for i in range(5):
        nm_result = minimize(target,prices_init, method='Nelder-Mead')

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
    

    
    
