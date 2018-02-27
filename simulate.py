import numpy as np
import time
import subprocess
from SCEconomy_give_A import Economy

import pickle


if __name__ == '__main__':

    def curvedspace(begin, end, curve, num=100):
        import numpy as np
        ans = np.linspace(0, (end - begin)**(1.0/curve), num) ** (curve) + begin
        ans[-1] = end #so that the last element is exactly end
        return ans


    ### additional info
#    agrid2 = curvedspace(0., 100., 2., 40)
#    kapgrid2 = curvedspace(0., 2., 2., 20)
#    zgrid2 = np.load('./input_data/zgrid.npy') ** 2.0

    agrid2 = curvedspace(0., 100., 2., 40)
   
    zgrid2 = np.load('./input_data/zgrid.npy') ** 2.
    
    


    ###define additional parameters###
    num_core = 8 #7 or 8 must be the best for Anmol's PC. set 3 or 4 for Yuki's laptop

    # prices
    w_ = 3.12149770513
    p_ = 0.963758557626
    rc_ = 0.0636523764656


    ###end defining additional parameters###

    print('Solving the model with the given prices...')
    print('Do not simulate more than one models at the same time...')

    econ = Economy(agrid = agrid2, zgrid = zgrid2, rho = 0.75)
   
    
    econ.set_prices(w = w_, p = p_, rc = rc_)
    with open('econ.pickle', mode='wb') as f: pickle.dump(econ, f)

    t0 = time.time()

    result = subprocess.run(['mpiexec', '-n', str(num_core), 'python', 'SCEconomy_give_A.py'], stdout=subprocess.PIPE)
    
    t1 = time.time()

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

    
    
    econ.calc_moments()
    ###calculate other important variables###
    econ.calc_sweat_eq_value()
    econ.calc_age()
    econ.simulate_other_vars()
    econ.save_result()
    

    
    
