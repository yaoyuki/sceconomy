
import sys
args = sys.argv

import numpy as np
import time
import subprocess
### use modified version of SCEConomy module
from SCEconomy_bizluv_give_A import Economy

import pickle
import numba as nb

w_init = float(args[1])
p_init = float(args[2])
rc_init = float(args[3])
num_core = args[4]

print('the code is running with ', num_core, 'cores...')
prices_init = [w_init, p_init, rc_init]


nd_log_file = '/home/ec2-user/Dropbox/sceconomy/log.txt'
detailed_output_file = '/home/ec2-user/Dropbox/sceconomy/detailed_output.txt'

f = open(detailed_output_file, 'w')
f.close()

dist_min = 10000000.0
econ_save = None

def curvedspace(begin, end, curve, num=100):
    import numpy as np
    ans = np.linspace(0, (end - begin)**(1.0/curve), num) ** (curve) + begin
    ans[-1] = end #so that the last element is exactly end
    return ans

agrid2 = curvedspace(0., 100., 2., 40)
# kapgrid2 = curvedspace(0., 1., 2., 20)
zgrid2 = np.load('./input_data/zgrid.npy') ** 2.

#determine bgrid and prob_
bgrid2 = np.array([0., 0.1])
prob_b = np.array([[0.8, 0.2], [0.2, 0.8]])



prob_base = np.load('./input_data/transition_matrix.npy')
prob2 = np.kron(prob_b, prob_base)

###codes to generate shocks###
num_s = prob2.shape[0]
num_pop_assigned = 100_000
sim_time = 2000
data_i_s_elem = np.ones((num_pop_assigned, sim_time), dtype = int) * (-1)

data_i_s_elem[:, 0] = 7

@nb.jit(nopython = True)
def transit(i, r):

    if r <= prob2[i,0]:
        return 0

    for j in range(1, num_s):

        #print(np.sum(prob[i,0:j]))
        if r <= np.sum(prob2[i,0:j]):
            return j - 1

    if r > np.sum(prob2[i,0:-1]) and r <= 1.:
        return num_s - 1

    print('error')

    return -1

np.random.seed(0) #fix the seed
data_rnd = np.random.rand(num_pop_assigned, sim_time)

@nb.jit(nopython = True, parallel = True)
def calc_trans(data_i_s_):
    for i in nb.prange(num_pop_assigned):
        for t in range(1, sim_time):
            data_i_s_[i, t] = transit(data_i_s_[i, t-1], data_rnd[i, t])
calc_trans(data_i_s_elem)

np.save('./input_data/data_i_s_bizluv.npy', data_i_s_elem) #save this result
###end codes to generate shocks###





def target(prices):
    global dist_min
    global econ_save
    
    w_ = prices[0]
    p_ = prices[1]
    rc_ = prices[2]
    
    print('computing for the case w = {:f}, p = {:f}, rc = {:f}'.format(w_, p_, rc_), end = ', ')
    
    ###set any additional condition/parameters
    ### alpha = 0.4 as default, and nu = 1. - phi - alpha
    econ = Economy(agrid = agrid2, zgrid = zgrid2, bgrid = bgrid2 , prob = prob2)


    econ.set_prices(w = w_, p = p_, rc = rc_)
    
    with open('econ.pickle', mode='wb') as f: pickle.dump(econ, f)
    #with open('econ.pickle', mode='rb') as f: econ = pickle.load(f)
    t0 = time.time()

    result = subprocess.run(['mpiexec', '-n', num_core, 'python', 'SCEconomy_bizluv_give_A.py'], stdout=subprocess.PIPE)
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
    

    
    
