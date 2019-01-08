import numpy as np
import time
import subprocess
from SCEconomy_hy_ns import Economy, split_shock

import pickle


if __name__ == '__main__':

    

    def curvedspace(begin, end, curve, num=100):
        import numpy as np
        ans = np.linspace(0, (end - begin)**(1.0/curve), num) ** (curve) + begin
        ans[-1] = end #so that the last element is exactly end
        return ans


    # additional info
    agrid2 = curvedspace(0., 200., 2., 40)
    kapgrid2 = curvedspace(0., 2., 2., 20)
    zgrid2 = np.load('./input_data/zgrid.npy') ** 2.0

    ###define additional parameters###
    # num_core = 119 #crash at 119
    num_core = 4    
    # prices
    p_, rc_ = 1.3444481511528166, 0.061488058639202293


    split_shock('./input_data/data_i_s', 100_000, num_core)


    
    ###end defining additional parameters###

    print('Solving the model with the given prices...')
    print('Do not simulate more than one models at the same time...')

    econ = Economy(agrid = agrid2, kapgrid = kapgrid2, zgrid = zgrid2,  path_to_data_i_s = './input_data/data_i_s')
    
    econ.set_prices(p = p_, rc = rc_)
    with open('econ.pickle', mode='wb') as f: pickle.dump(econ, f)

    t0 = time.time()

    #don't forget to replace import argument
    result = subprocess.run(['mpiexec', '-n', str(num_core), 'python', 'SCEconomy_hy_ns.py'], stdout=subprocess.PIPE)
    
    t1 = time.time()

    detailed_output_file = './log/test.txt'
    f = open(detailed_output_file, 'ab') #use byte mode
    f.write(result.stdout)
    f.close()


    with open('econ.pickle', mode='rb') as f: econ = pickle.load(f)


    p = econ.p
    rc = econ.rc
    moms = econ.moms
    
    dist = np.sqrt(moms[0]**2.0 + moms[1]**2.0)
    
    if p != p_ or  rc != rc_:
        print('err: input prices and output prices do not coincide.')
        print('p = ', p, ', p_ = ', p_)
        print('rc = ', rc, ', rc_ = ', rc_)

    
    
    #calc main moments
    econ.calc_moments()
    
    ###calculate other important variables###
    #econ.calc_sweat_eq_value()
    #econ.calc_age()
    #econ.simulate_other_vars()
    #econ.save_result()
    

    
    
