import sys
args = sys.argv

import numpy as np
import time
import subprocess
### use modified version of SCEConomy module
from SCEconomy_LSC_ns_lifecycle import Economy, split_shock

import pickle

p_init = float(args[1])
rc_init = float(args[2])
#ome_init = float(args[3])
#nu_init = float(args[4])
num_core = int(args[3])


ome_init = 0.4561128052733918
# varpi_init = 0.559952019068588
theta_init = 0.5071181286751945


ome_ = ome_init
# varpi_ =  varpi_init
theta_ =  theta_init


print('the code is running with ', num_core, 'cores...')
# prices_init = [p_init, rc_init, ome_init, nu_init]
prices_init = [p_init, rc_init]

nd_log_file = '/cluster/shared/yaoxx366/log2/log.txt'
detailed_output_file = '/cluster/shared/yaoxx366/log2/detail.txt'


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


from markov import calc_trans, Stationary
#generate shock sequene
path_to_data_i_s = './tmp/data_i_s'
path_to_data_is_o = './tmp/data_is_o'    
num_pop = 100_000
sim_time = 3_000

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

GDP_guess = 3.20

taup = 0.36 *(1.0 - 0.278)
taub = np.array([0.137, 0.185, 0.202, 0.238, 0.266, 0.28]) *(1.0 -  0.506) #large one
psib = np.array([-0.010393600000000013, 0.012646399999999981, 0.03, 0.12492479999999993, 0.3117408000000001,0.4430048000000002])

taun = np.array([0.293,  0.317, 0.324,  0.343,  0.39,  0.405, 0.408,  0.419])
psin = np.array([-0.10037472000000003, -0.08685792000000002, -0.08193888000000002, -0.06546208, 0.0011951999999999727, 0.03,  0.04398335999999975, 0.14192735999999984])


def target(prices):
    global dist_min
    global econ_save

    p_ = prices[0]
    rc_ = prices[1]
    # ome_ = prices[2]
    # nu_ = prices[3]

    
    print('computing for the case p = {:f}, rc = {:f}'.format(p_, rc_), end = ', ')
    
    ###set any additional condition/parameters

    econ = Economy(sim_time = 1000, num_total_pop = num_pop,
                   agrid = agrid2,  zgrid = zgrid2, rho = 0.01, prob = prob,
                   ome = ome_, theta = theta_,
                   path_to_data_i_s = path_to_data_i_s, path_to_data_is_o = path_to_data_is_o,
                   scaling_n = GDP_guess, scaling_b = GDP_guess, g = 0.133*GDP_guess, yn = 0.266*GDP_guess, xnb = 0.110*GDP_guess,
                   delk = 0.041, delkap = 0.041,  veps = 0.418, vthet = 1.0 - 0.418,
                   tauc = 0.065, taud = 0.133,
                   taup = taup, taub = taub , psib = psib
                   #, epsgrid = epsgrid2
    )
    

    econ.set_prices(p = p_, rc = rc_)
    
    with open('econ.pickle', mode='wb') as f: pickle.dump(econ, f)
    #with open('econ.pickle', mode='rb') as f: econ = pickle.load(f)
    t0 = time.time()
    result = subprocess.run(['mpiexec', '-n', str(num_core), 'python', 'SCEconomy_LSC_ns_lifecycle.py'], stdout=subprocess.PIPE)
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
    ome = econ.ome
    theta = econ.theta
    moms = econ.moms
    

    dist = np.sqrt(moms[0]**2.0 + moms[1]**2.0)
    # dist = np.sqrt(moms[0]**2.0 + moms[1]**2.0 + (moms[4]/s_emp_share - 1.)**2.0 + (moms[5]/pure_sweat_share - 1.)**2.0)
    
    if p != p_ or  rc != rc_ or ome != ome_ or theta != theta_:
        print('err: input prices and output prices do not coincide.')
        print('p = ', p, ', p_ = ', p_)
        print('rc = ', rc, ', rc_ = ', rc_)
        print('ome = ', ome, ', ome_ = ', ome_)
        print('theta = ', theta, ', theta_ = ', theta_)                
       # return
    
    print('dist = {:f}'.format(dist))

    f = open(nd_log_file, 'a')
    f.writelines(str(p) + ', ' + str(rc) + ', ' + str(ome) + ', ' + str(theta) + ', ' + str(dist) + ', ' +  str(moms[0]) + ', ' + str(moms[1]) + ', ' + str(moms[2]) + ', ' + str(moms[3]) + '\n')
    f.close()
    
    if dist < dist_min:
        econ_save = econ
        dist_min = dist
    return dist

if __name__ == '__main__':

    f = open(nd_log_file, 'w')
    f.writelines('p, rc, ome, theta, dist, mom0, mom1, mom2, mom3\n')
    f.writelines('GDP_guess = ' +  str(GDP_guess) + '\n')        
    f.close()

    ### generate shocks ###
    np.random.seed(0)
    data_rand = np.random.rand(num_pop, sim_time)
    data_i_s = np.ones((num_pop, sim_time), dtype = int)
    data_i_s[:, 0] = 7 #initial state. it does not matter if simulation is long enough.
    calc_trans(data_i_s, data_rand, prob)
    data_i_s = data_i_s[:, 2000:]
    np.save(path_to_data_i_s + '.npy' , data_i_s)
    split_shock(path_to_data_i_s, num_pop, num_core)
    del data_rand
    

    np.random.seed(2)
    data_rand = np.random.rand(num_pop, sim_time+1) #+1 is added since this matters in calculation
    data_is_o = np.ones((num_pop, sim_time+1), dtype = int)
    data_is_o[:, 0] = 0 #initial state. it does not matter if simulation is long enough.
    calc_trans(data_is_o, data_rand, prob_yo)
    data_is_o = data_is_o[:, 2000:]
    np.save(path_to_data_is_o + '.npy' , data_is_o)
    split_shock(path_to_data_is_o, num_pop, num_core)
    del data_rand

    ### end generate shocks ###




    
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
    

    
    
