
#import Yuki's library in the directory ./library
import sys
sys.path.insert(0, './library/')


import numpy as np
import numba as nb
###usage
###@nb.jit(nopython = True)


#my library
#import
from orderedTableSearch import locate, hunt
from FEM import femeval, fem_peval #1D interpolation
from FEM_2D import fem2d_peval, fem2deval_mesh #2D interpolation
from markov import Stationary
from ravel_unravel_nb import unravel_index_nb


from mpi4py import MPI
comm = MPI.COMM_WORLD #retreive the communicator module
rank = comm.Get_rank() #get the rank of the process
size = comm.Get_size() #get the number of processes

import time



# %matplotlib inline
# import matplotlib as mpl
# mpl.rc("savefig",dpi=100)
# from matplotlib import pyplot as plt




class Economy:
    """
    class Economy stores all the economic parameters, prices, computational parameters, 
    grids information for a, kappa, eps, z, and shock transition matrix
    """
    
    def __init__(self,
                 alpha = None,
                 nu = None,
                 beta = None,
                 iota = None,
                 chi = None,
                 delk = None,
                 delkap = None,
                 eta = None,
                 g = None,
                 grate = None,
                 la = None,

                 tau_wo = None,
                 tau_bo = None,
                 
                 mu = None,
                 ome = None,
                 phi = None,
                 rho = None,
                 tauc = None,
                 taud = None,
                 taup = None,
                 theta = None,
                 trans_retire = None,
                 veps = None,
                 vthet = None,
                 xnb = None,
                 yn = None,
                 zeta = None,
                 lbar = None,
                 agrid = None,
                 epsgrid = None,
                 zgrid = None,
                 prob = None,
                 prob_yo = None,
                 
                 is_to_iz = None,
                 is_to_ieps = None,
                 amin = None,
                 sim_time = None,
                 num_total_pop = None,
                 A = None,
                 path_to_data_i_s = None,
                 path_to_data_is_o = None,

                 taun = None,
                 psin = None,
                 psin_fixed = None,
                 nbracket = None,
                 nbracket_fixed = None,                 
                 scaling_n = None,
                 
                 taub = None,
                 psib = None,
                 psib_fixed = None,                 
                 bbracket = None,
                 bbracket_fixed = None,                 
                 scaling_b = None):

        
        self.__set_default_parameters__()
        
        #set the parameters if designated
        #I don't know how to automate these lines
        if alpha is not None: self.alpha = alpha
        if nu is not None: self.nu = nu
        if beta is not None: self.beta = beta
        if iota is not None: self.iota = iota
        if chi is not None: self.chi = chi
        if delk is not None: self.delk = delk    
        if delkap is not None: self.delkap = delkap
        if eta is not None: self.eta = eta
        if g is not None: self.g = g
        if grate is not None: self.grate = grate 
        if la is not None: self.la = la

        if tau_wo is not None: self.tau_wo = tau_wo
        if tau_bo is not None: self.tau_bo = tau_bo
        
        if mu is not None: self.mu = mu 
        if ome is not None: self.ome = ome 
        if phi is not None: self.phi = phi
        if rho is not None: self.rho = rho
        if tauc is not None: self.tauc = tauc
        if taud is not None: self.taud = taud
#         if taum is not None: self.taum = taum
#         if taun is not None: self.taun = taun
        if taup is not None: self.taup = taup
        if theta is not None: self.theta = theta
        if trans_retire is not None: self.trans_retire = trans_retire #added        
        if veps is not None: self.veps = veps
        if vthet is not None: self.vthet = vthet
        if xnb is not None: self.xnb = xnb
        if yn is not None: self.yn = yn
        if zeta is not None: self.zeta = zeta
        if lbar is not None: self.lbar = lbar
        if agrid is not None: self.agrid = agrid
        if epsgrid is not None: self.epsgrid = epsgrid
        if zgrid is not None: self.zgrid = zgrid
        if prob is not None: self.prob = prob
        if prob_yo is not None: self.prob_yo = prob_yo        
        if is_to_iz is not None: self.is_to_iz = is_to_iz
        if is_to_ieps is not None: self.is_to_ieps = is_to_ieps
        if amin is not None: self.amin = amin
        if sim_time is not None: self.sim_time = sim_time
        if num_total_pop is not None: self.num_total_pop = num_total_pop
        if A is not None: self.A = A

        if path_to_data_i_s is not None: self.path_to_data_i_s = path_to_data_i_s
        if path_to_data_is_o is not None: self.path_to_data_is_o = path_to_data_is_o
        

        if taun is not None: self.taun = taun
        if psin is not None: self.psin = psin
        if psin_fixed is not None: self.psin_fixed = psin_fixed
        if nbracket is not None: self.nbracket = nbracket
        if nbracket_fixed is not None: self.nbracket_fixed = nbracket_fixed                
        if scaling_n is not None: self.scaling_n = scaling_n

        if taub is not None: self.taub = taub
        if psib is not None: self.psib = psib
        if psib_fixed is not None: self.psib_fixed = psib_fixed        
        if bbracket is not None: self.bbracket = bbracket
        if bbracket_fixed is not None: self.bbracket_fixed = bbracket_fixed  
        if scaling_b is not None: self.scaling_b = scaling_b

        #c
        # if self.upsilon >= 1.0:
        #     print('Error: upsilon must be < 1 but upsilon = ', upsilon)

        self.__set_nltax_parameters__()
        self.__set_implied_parameters__()
        
    
    def __set_default_parameters__(self):
        """
        Load the baseline value
        """
        
        self.__is_price_set__ = False
        self.alpha    = 0.3
        self.nu       = 0.55
        self.beta     = 0.98
        self.iota     = 1.0
        self.chi      = 0.0 #param for borrowing constarint
        self.delk     = 0.05
        self.delkap   = 0.05 
        self.eta      = 0.42
        self.g        = 0.234 #govt spending
        self.grate    = 0.02 #gamma, growth rate for detrending
        self.la       = np.inf #lambda
        self.tau_wo   = 0.5 #added
        self.tau_bo   = 0.5 #added
        self.mu       = 1.5 
        self.ome      = 0.6 #omega
        self.phi      = np.nan
        self.rho      = 0.01
        self.tauc     = 0.06
        self.taud     = 0.14
        self.taum     = 0.20
        self.taun     = 0.40
        self.taup     = 0.30
        self.theta    = 0.41
        self.trans_retire = 0.48        
        self.veps     = 0.4
        self.vthet    = 0.4
        self.xnb      = 0.185
        self.yn       = 0.451
        self.zeta     = 1.0 #totally tentative
        self.lbar     = 1.0

        #borrowing constraint for C - guys. they just can't borrow in
        self.amin      = 0.0
        
        self.sim_time = 1_000
        self.num_total_pop = 100_000
        self.A        = 1.577707121233179 #this should give yc = 1 (approx.) z^2 case

        self.path_to_data_i_s = './tmp/data_i_s'
        self.path_to_data_is_o = './tmp/data_is_o'


        self.taub = np.array([.137, .185, .202, .238, .266, .280])
        self.bbracket = np.array([0.150, 0.319, 0.824, 2.085, 2.930])
        self.scaling_b = 1.0
        self.psib_fixed = 0.03
        self.bbracket_fixed = 2
        self.psib = None

        self.taun = np.array([.2930, .3170, .3240, .3430, .3900, .4050, .4080, .4190])
        self.nbracket = np.array([.1760, .2196, .2710, .4432, 0.6001, 1.4566, 2.7825])
        self.scaling_n = 1.0
        self.psin_fixed = 0.03
        self.nbracket_fixed = 5
        self.psin = None         

        

        self.agrid = np.load('./input_data/agrid.npy')
        self.epsgrid = np.load('./input_data/epsgrid.npy')    
        self.zgrid = np.load('./input_data/zgrid.npy')
        

        #conbined exogenous states
        #s = (e,z)'

        #pi(t,t+1)
        self.prob = np.load('./DeBacker/prob_epsz.npy') #default transition is taken from DeBakcer
        self.prob_yo = np.array([[44./45., 1./45.], [3./45., 42./45.]]) #[[y -> y, y -> o], [o -> y, o ->o]]
        
    

        # ####do we need this one here?
        # #normalization to correct rounding error.
        # for i in range(prob.shape[0]):
        #     prob[i,:] = prob[i,:] / np.sum(prob[i,:])

        self.is_to_iz = np.load('./input_data/is_to_iz.npy')
        self.is_to_ieps = np.load('./input_data/is_to_ieps.npy')


        self.s_age = None
        self.c_age = None
        self.sind_age = None
        self.cind_age = None
        self.y_age = None
        self.o_age = None
        

    def __set_nltax_parameters__(self):
        
        from LinearTax import get_consistent_phi #the name is wrong

        
        self.bbracket = self.bbracket * self.scaling_b
        #set transfer term if not provided
        if self.psib is None:
            self.psib = get_consistent_phi(self.bbracket, self.taub, self.psib_fixed, self.bbracket_fixed) # we need to set the last two as arguments

        tmp = self.bbracket
        self.bbracket = np.zeros(len(tmp)+2)

        self.bbracket[0] = -np.inf
        self.bbracket[-1] = np.inf
        self.bbracket[1:-1] = tmp[:]

        
        
        self.nbracket = self.nbracket * self.scaling_n

        #set transfer term if not provided
        if self.psin is None:
            self.psin = get_consistent_phi(self.nbracket, self.taun, self.psin_fixed, self.nbracket_fixed) # we need to set the last two as arguments

        tmp = self.nbracket
        self.nbracket = np.zeros(len(tmp)+2)        
        self.nbracket[0] = -np.inf
        self.nbracket[-1] = np.inf
        self.nbracket[1:-1] = tmp[:]
       

        
    def __set_implied_parameters__(self):
        #length of grids
        self.num_a = len(self.agrid)
        self.num_eps = len(self.epsgrid)
        self.num_z = len(self.zgrid)
        self.num_s = self.prob.shape[0]

        
        #implied parameters
        self.bh = self.beta*(1. + self.grate)**(self.eta*(1. - self.mu))  #must be less than one.
        self.varrho = (1. - self.alpha - self.nu)/(1. - self.alpha) * self.vthet / (self.vthet + self.veps)
    
        if self.bh >= 1.0:
            print('Error: bh must be in (0, 1) but bh = ', self.bh)

        self.prob_st = Stationary(self.prob)
        self.prob_yo_st = Stationary(self.prob_yo)            
        
        
        
    def set_prices(self, p, rc):
        self.p = p
        self.rc = rc

        #assuming CRS technology for C-corp
        self.kcnc_ratio = ((self.theta * self.A)/(self.delk + self.rc))**(1./(1. - self.theta))
        self.w = (1. - self.theta)*self.A*self.kcnc_ratio**self.theta
        
        self.__is_price_set__ = True
        
        
        #implied prices
        self.rbar = (1. - self.taup) * self.rc
        self.rs = (1. - self.taup) * self.rc

        self.xi1 = ((self.ome*self.p)/(1. - self.ome))**(1./(self.rho-1.0)) #ok
        self.xi2 = (self.ome + (1. - self.ome) * self.xi1**self.rho)**(1./self.rho) #ok

        self.xi3 = self.eta/(1. - self.eta) * self.ome / (1. + self.tauc) / self.xi2**self.rho #changed

        self.xi8 = ((self.p*(self.nu**self.nu)*(self.alpha**(1.-self.nu))/((self.w**self.nu)*((self.rs + self.delk)**(1.-self.nu)) )))**(1./(1.-self.alpha-self.nu))
                
        self.xi13 = self.nu/self.alpha*(self.rs + self.delk)/self.w #ns = xi13*ks


        self.denom = (1. + self.p*self.xi1)*(1. + self.tauc)
        self.xi7 = 1./ self.denom #changed        
        self.xi4 = (1. + self.rbar) / self.denom #ok
        self.xi5 = (1. + self.grate) / self.denom #ok
        self.xi6 = (self.yn - self.xnb) / self.denom #changed
        


    def print_parameters(self):

        print('')
        print('Parameters')
        print('alpha = ', self.alpha)
        print('nu = ', self.nu)        
        print('beta = ', self.beta)
        print('chi = ', self.chi)
        print('delk = ', self.delk)
        print('delkap = ', self.delkap)
        print('eta = ', self.eta)
        print('g (govt spending) = ', self.g)
        print('grate (growth rate of the economy) = ', self.grate)
        print('la = ', self.la)
        print('mu = ', self.mu)
        print('ome = ', self.ome)
        print('phi = ', self.phi)
        print('rho = ', self.rho)
        # print('varpi = ', self.varpi)
        print('tauc = ', self.tauc)
        print('taud = ', self.taud)
        print('taup = ', self.taup)
        print('theta = ', self.theta)

        print('veps = ', self.veps)
        print('vthet = ', self.vthet)
        print('xnb = ', self.xnb)
        print('yn = ', self.yn)
        print('zeta = ', self.zeta)
        print('A = ', self.A)


        print('')
        print('nonlinear tax function')


        for ib, tmp in enumerate(self.taub):
            print(f'taub{ib} = {tmp}')
        for ib, tmp in enumerate(self.psib):
            print(f'psib{ib} = {tmp}')            
        for ib, tmp in enumerate(self.bbracket):
            print(f'bbracket{ib} = {tmp}')
        for i, tmp in enumerate(self.taun):
            print(f'taun{i} = {tmp}')
        for i, tmp in enumerate(self.psin):
            print(f'psin{i} = {tmp}')            
        for i, tmp in enumerate(self.nbracket):
            print(f'nbracket{i} = {tmp}')

        print('')
        print('Parameters specific to a lifecycle model')
        print('iota = ', self.iota) #added
        # print('la_tilde = ', self.la_tilde) #added
        print('tau_wo = ', self.tau_wo) #added
        print('tau_bo = ', self.tau_bo) #added
        print('trans_retire = ', self.trans_retire)
        
        print(f'prob_yo =  {self.prob_yo[0,0]}, {self.prob_yo[0,1]}, {self.prob_yo[1,0]}, {self.prob_yo[1,1]}.') #added
        print('statinary dist of prob_yo = ', self.prob_yo_st) #added
        print('')
            

        
        
        
        if self.__is_price_set__:
            
            print('')
            print('Prices')
            print('w = ', self.w)
            print('p = ', self.p)
            print('rc = ', self.rc)
            print('')
            print('Implied prices')
            print('rbar = ', self.rbar)
            print('rs = ', self.rs)

            print('')
            print('')
            print('')


            print('')
            print('xi1 = ', self.xi1)
            print('xi2 = ', self.xi2)
            print('xi3 = ', self.xi3)
            print('xi4 = ', self.xi4)
            print('xi5 = ', self.xi5)
            print('xi6 = ', self.xi6)
            print('xi7 = ', self.xi7)
            print('xi8 = ', self.xi8)
            print('xi13 = ', self.xi13)            

            
        else:
            print('')
            print('Prices not set')


        print('')
        print('Computational Parameters')
        print('amin = ', self.amin)
        print('sim_time = ', self.sim_time)
        print('num_total_pop = ', self.num_total_pop)
            
        
    def generate_util(self):

        bh = self.bh
        eta = self.eta
        mu = self.mu
        
        @nb.jit(nopython = True)
        def util(c, l):
            if c > 0.0 and l > 0.0 and l <= 1.0:
                return (1. - bh) * (((c**eta)*(l**(1. - eta)))**(1. - mu))
            else:
                if mu < 1.0:
                    return -np.inf
                else:
                    return np.inf
                #not way to code mu = 1.0 case

        return util
    
    def generate_dc_util(self):

        bh = self.bh
        eta = self.eta
        mu = self.mu
        
        #this is in the original form
        @nb.jit(nopython = True)
        def dc_util(c, l):
            if c > 0.0 and l > 0.0 and l <= 1.0:
                return eta * c**(eta*(1. - mu) - 1.0) * ((l**(1. - eta)))**(1. - mu)

            else:
                print('dc_util at c = ', c, ', l = ', l, 'is not defined.')
                print('nan will be returned.')
                return np.nan #???
            
        return dc_util

     # import math

    def generate_cstatic(self):

        # for variable in self.__dict__ : exec(variable+'= self.'+variable)

        alpha = self.alpha
        beta = self.beta
        chi = self.chi
        delk = self.delk
        delkap = self.delkap
        eta = self.eta
        g = self.g
        grate = self.grate
        la = self.la
        mu = self.mu
        ome = self.ome
        phi = self.phi
        rho = self.rho
        tauc = self.tauc
        taud = self.taud
        taup = self.taup
        theta = self.theta
        veps = self.veps
        vthet = self.vthet
        xnb = self.xnb
        yn = self.yn
        zeta = self.zeta

        tau_wo = self.tau_wo
        trans_retire = self.trans_retire

        taun = self.taun
        psin = self.psin
        nbracket = self.nbracket

        agrid = self.agrid
        epsgrid = self.epsgrid
        zgrid = self.zgrid

        prob = self.prob

        is_to_iz = self.is_to_iz
        is_to_ieps = self.is_to_ieps

        amin = self.amin

        num_a = self.num_a
        num_eps = self.num_eps
        num_z = self.num_z

        nu = self.nu
        bh = self.bh

        w = self.w
        p = self.p
        rc = self.rc

        rbar = self.rbar
        rs = self.rs

        xi1 = self.xi1
        xi2 = self.xi2
        xi3 = self.xi3
        xi4 = self.xi4
        xi5 = self.xi5
        xi6 = self.xi6
        xi7 = self.xi7
        xi8 = self.xi8
        
        util = self.generate_util()
            
        @nb.jit(nopython = True)
        def get_cstatic(s):
            a = s[0]
            an = s[1]
            eps = s[2]
            is_o = s[3]

            u = -np.inf
            cc = -1.0
            cs = -1.0
            cagg = -1.0

            l = -1.0
            n = -1.0

            if is_o:
                eps = tau_wo*eps #replace eps with tau_wo*eps

            #is this unique?
            #repeat until n falls in bracket nuber i (i=0,1,2,..,I-1)
            i = 0
            j = 0
            num_taun = len(taun)
            wepsn = 0.0
            for i in range(num_taun):
                n = (xi3*w*eps*(1.-taun[i]) - xi4*a + xi5*an - xi6 - xi7*(psin[i] + is_o*trans_retire))/(w*eps*(1.-taun[i])*(xi3 + xi7))
                # n = (xi3*w*eps*(1.-taun[i]) - xi4*a + xi5*an - xi6 - xi7*(psin[i] + is_o*trans_retire))/(w*eps*(1.-taun[i])*(xi3 + xi7))
                
                wepsn = w*eps*n #wageincome
                j = locate(wepsn, nbracket)

                if i == j:
                    break

            obj_i = 0.
            obj_i1 = 0.
            #when solution is at a kink
            flag = True
            flag2 = False
            
            if i == len(taun) - 1 and i != j:
                flag = False
                flag2 = True

                for i, wepsn in enumerate(nbracket[1:-1]): #remove -inf, inf
                    n = wepsn/w/eps

                    obj_i = n - ( (xi3*w*eps*(1.-taun[i]) - xi4*a + xi5*an - xi6 - xi7*(psin[i]+is_o*trans_retire))/(w*eps*(1.-taun[i])*(xi3 + xi7)))
                    obj_i1 = n - ( (xi3*w*eps*(1.-taun[i+1]) - xi4*a + xi5*an - xi6 - xi7*(psin[i+1]+is_o*trans_retire))/(w*eps*(1.-taun[i+1])*(xi3 + xi7)))                    
                    
                    # obj_i = n - ( (xi3*w*eps*(1.-taun[i]) - xi4*a + xi5*an - xi6 - xi7*(psin[i] - is_o*trans_retire))/(w*eps*(1.-taun[i])*(xi3 + xi7)))
                    # obj_i1 = n - ( (xi3*w*eps*(1.-taun[i+1]) - xi4*a + xi5*an - xi6 - xi7*(psin[i+1] - is_o*trans_retire))/(w*eps*(1.-taun[i+1])*(xi3 + xi7)))

                    if obj_i * obj_i1 < 0:
                        flag = True
                        break
                    
            # if flag2 and flag:
            #     print('a solution is found at a corner ')
            #     print('obj_i = ', obj_i)
            #     print('obj_i1 = ', obj_i1)
            #     print('a = ', a)
            #     print('an = ', an)
            #     print('eps = ', eps)
            #     print('i = ', i)

            #     # print('j = ', j)
            #     print('n = ', n)
            #     print('wepsn = ', wepsn)
            #     print('')

            if not flag:
                print('no solution find ')
                print('err: cstatic: no bracket for n')
                print('a = ', a)
                print('an = ', an)
                print('eps = ', eps)
                print('is_o = ', is_o)
                print('trans_retire = ', trans_retire)
                print('i = ', i)
                print('j = ', j)
                print('n = ', n)
                print('w = ', w)
                print('wepsn = ', wepsn)
                print('')
                                        

            if n < 0.0:
                n = 0.0
                wepsn = w*eps*n
                i = locate(wepsn, nbracket)
                

            if n >= 0. and n <= 1.:

                l = 1. - n

                #cc from FOC  is wrong at the corner.
                cc = xi4*a - xi5*an + xi6 + xi7*((1.-taun[i])*w*eps*n + psin[i] + is_o*trans_retire)
                cs = xi1*cc
                cagg = xi2*cc
                u = util(cagg, 1. - n)


            return u, cc, cs, cagg, l ,n, -1.0e23, -1.0e23, i, taun[i], psin[i] + is_o*trans_retire

    
        return get_cstatic

    
    
    def generate_sstatic(self):
        ###load vars###
        alpha = self.alpha
        beta = self.beta
        chi = self.chi
        delk = self.delk
        delkap = self.delkap
        eta = self.eta
        g = self.g
        grate = self.grate
        la = self.la
        mu = self.mu
        ome = self.ome
        phi = self.phi
        rho = self.rho
        tauc = self.tauc
        taud = self.taud
        taup = self.taup
        theta = self.theta
        veps = self.veps
        vthet = self.vthet
        xnb = self.xnb
        yn = self.yn
        zeta= self.zeta
        lbar = self.lbar

        tau_bo = self.tau_bo
        trans_retire = self.trans_retire
        
        taub = self.taub
        psib = self.psib
        bbracket = self.bbracket

        agrid = self.agrid
        epsgrid = self.epsgrid
        zgrid = self.zgrid

        prob = self.prob

        is_to_iz = self.is_to_iz
        is_to_ieps = self.is_to_ieps

        amin = self.amin

        num_a = self.num_a
        num_eps = self.num_eps
        num_z = self.num_z

        nu = self.nu
        bh = self.bh
        varrho = self.varrho

        w = self.w
        p = self.p
        rc = self.rc

        rbar = self.rbar
        rs = self.rs
    
        denom = self.denom
        xi1 = self.xi1
        xi2 = self.xi2
        xi3 = self.xi3
        xi4 = self.xi4
        xi5 = self.xi5
        xi6 = self.xi6
        xi7 = self.xi7
        xi8 = self.xi8
        xi13 = self.xi13        
        ###end loading vars###
        
        util = self.generate_util()

        @nb.jit(nopython = True)
        def get_sstatic(s):

            a = s[0]
            an = s[1]
            z = s[2]
            is_o = s[3]

            if is_o:
                z = tau_bo*z #replace eps with tau_wo*eps
            
            # ks = (z*p*alpha/(rs+delk))**(1./(1. - alpha))
            
            ks = z**(1./(1. - alpha - nu))*xi8 #should be the same as above
            ns = xi13*ks

            if an < chi *ks: #if the working capital constraint is binding

                if an > 0.:
                    ks = an / chi
                    ns = (nu*p*z*ks**alpha/w)**(1./(1.0 - nu))
                else:
                    ks = 0.0
                    ns = 0.0
            
            ys = z*(ks**alpha)*(ns**nu)

            #bizinc does not depend on tax rate
            bizinc = p*ys - (rs+delk)*ks - w*ns
            ibracket = locate(bizinc, bbracket) #check if this is actually working


            #initial
            u = -np.inf
            cc = -1.0
            cs = -1.0
            cagg = -1.0

            
            # cc = xi4*a - xi5*an + xi6 + xi11*(p*ys - (rs+delk)*ks)
            cc = xi4*a - xi5*an + (1.- taub[ibracket])/denom*bizinc + (psib[ibracket] + is_o*trans_retire + yn - xnb)/denom
            # cc = xi4*a - xi5*an + (1.- taub[ibracket])*xi7*bizinc + (psib[ibracket])*xi7 + xi6
            cs = xi1*cc
            cagg = xi2 * cc

            #adhoc feasbility check
            if (cagg > 0.0):
                u = util(cagg, lbar)
                
            #else, return -np.inf


            # lbar is a given constant
            #mx, my, x are set to np.nan.
            return u, cc, cs, cagg, lbar, -1.0e20, ks, ys, ibracket, taub[ibracket], psib[ibracket] + is_o*trans_retire, ns
        
        return get_sstatic


        
    def get_policy(self, max_iter = 20, max_howard_iter = 100):
        # I found this magic is very dangerous.
        for variable in self.__dict__ : exec(variable+'= self.'+variable, locals(), globals())        

        Econ = self
        num_total_state = num_a* num_s
        m = num_total_state // size
        r = num_total_state % size

        assigned_state_range = (rank*m+min(rank,r),(rank+1)*m+min(rank+1,r))
        num_assigned = assigned_state_range[1] - assigned_state_range[0]


        all_assigned_state_range = np.ones((size, 2))*(-2.)
        all_num_assigned = ()
        all_istart_assigned = ()


        for irank in range(size):

            all_istart_assigned += (int(irank*m+min(irank,r)), )
            all_assigned_state_range[irank,0] = irank*m+min(irank,r)
            all_assigned_state_range[irank,1] = (irank+1)*m+min(irank+1,r)
            all_num_assigned += (int(all_assigned_state_range[irank,1] - all_assigned_state_range[irank,0]),) 

        ###end parameters for MPI###



        @nb.jit(nopython = True)
        def unravel_ip(i_aggregated_state):

            istate, ia = unravel_index_nb(i_aggregated_state, num_s, num_a)
            #ia, ikap, istate = unravel_index_nb(i_aggregated_state, num_a, num_kap, num_s)
            return istate, ia

        get_cstatic = Econ.generate_cstatic()


        c_supan = np.ones((num_a, num_eps, 2)) * (-2.)

        
        #for young c-corp workers
        for ia, a in enumerate(agrid):
                for ieps, eps in enumerate(epsgrid):

                    c_supan[ia, ieps, 0] = ((1. + rbar)*a + (1. - taun[0])*w*eps + psin[0] + yn - xnb)/(1. + grate)
                    
        #for old c-corp workers
        for ia, a in enumerate(agrid):
                for ieps, eps in enumerate(epsgrid):

                    c_supan[ia, ieps, 1] = ((1. + rbar)*a + (1. - taun[0])*tau_wo*w*eps + psin[0] + trans_retire + yn - xnb)/(1. + grate)
        

        get_sstatic = Econ.generate_sstatic()                    
        #to solve S-optimization problem, we need the max feasible set for an [amin, sup_an]                    

        s_supan = np.ones((num_a, num_z, 2)) * (-2.)
        
        ### ks = xi8  ##this is wrong

        for iz, z in enumerate(zgrid):
            for ia, a in enumerate(agrid):
                for is_old in range(2):
            
                    ks = z**(1./(1. - alpha - nu))*xi8 #should be the same as above
                    ns = xi13*ks            
                    ys = z*(ks**alpha)*(ns**nu)
                

                    #first tax bracket is picked. I don't have a clear reqson for it.
                    s_supan[ia, iz, is_old] = ((1. + rbar)*a + (1. - taub[0])*(p*ys - (rs + delk)*ks - w*ns) + psib[0] + is_old*trans_retire + yn - xnb)/(1. + grate)

        del ks, ys, ns
                    
        # objective function of VFI optimization problem
        @nb.jit(nopython = True)    
        def _obj_loop_(*args):
            _an_ = args[0]
            _EV_ = args[1] 
            _ia_ = args[2]
            _istate_ = args[3]
            _is_c_ = args[4]
            _is_old_ = args[5] # 0 (young) or 1 (old)

            u = 0.0

            if _is_c_:
                u = get_cstatic(np.array([agrid[_ia_], _an_, epsgrid[is_to_ieps[_istate_]], _is_old_]))[0]
            else:
                u = get_sstatic(np.array([agrid[_ia_], _an_, zgrid[is_to_iz[_istate_]], _is_old_]))[0]

            # isn't it wrong?
            # return -(u + bh*fem_peval(_an_, agrid,  _EV_[0, :, _istate_])**(1. - mu))**(1./(1. - mu))

            # usually _EV_ is already discounted            
            return -(u + fem_peval(_an_, agrid,  _EV_[0, :, _istate_] ))**(1./(1. - mu)) 
            
        #epsilon = np.finfo(float).eps
        @nb.jit(nopython = True)
        def _optimize_given_state_(_an_min_, _an_sup_, _EV_, _ia_ ,_istate_, _is_c_, _is_old_):        

            #arguments
            ax = _an_min_
            cx = _an_sup_
            bx = 0.5*(ax + cx)

            tol=1.0e-8
            itmax=500

            #parameters
            CGOLD=0.3819660
            ZEPS=1.0e-3*2.2204460492503131e-16
            #*np.finfo(float).eps

            brent = 1.0e20
            xmin = 1.0e20

            a=min(ax,cx)
            b=max(ax,cx)
            v=bx
            w=v
            x=v
            e=0.0

#            print('is_c = ',_is_c_)

            fx= _obj_loop_(x,  _EV_, _ia_ ,_istate_, _is_c_, _is_old_) 
            fv=fx
            fw=fx

            d = 0.0

            it = 0
            for it in range(itmax):


                xm=0.5*(a+b)
                tol1=tol*abs(x)+ZEPS
                tol2=2.0*tol1

                tmp1 = tol2 - 0.5*(b-a)
                tmp2 = abs(x-xm)



                if abs(x - xm) <= tol2 - 0.5*(b - a):
                    it = itmax

                    xmin=x
                    # brent=fx


                if (abs(e) > tol1):
                    r=(x-w)*(fx-fv)
                    q=(x-v)*(fx-fw)
                    p=(x-v)*q-(x-w)*r
                    q=2.0*(q-r)

                    if (q > 0.0): 
                        p=-p

                    q=abs(q)
                    etemp=e
                    e=d

                    if abs(p) >= abs(0.5*q*etemp) or  p <= q*(a-x) or p >= q*(b-x):

                        #e=merge(a-x,b-x, x >= xm )
                        if x >= xm:
                            e = a-x
                        else:
                            e = b-x
                        d=CGOLD*e

                    else:
                        d=p/q
                        u=x+d

                        if (u-a < tol2 or b-u < tol2): 
                            d= abs(tol1)*np.sign(xm - x)  #sign(tol1,xm-x)

                else:

                    if x >= xm:
                        e = a-x
                    else:
                        e = b-x

                    d=CGOLD*e

                u = 0.  #merge(x+d,x+sign(tol1,d), abs(d) >= tol1 )
                if abs(d) >= tol1:
                    u = x+d
                else:
                    u = x+abs(tol1)*np.sign(d)

                fu = _obj_loop_(u, _EV_, _ia_ ,_istate_, _is_c_, _is_old_)

                if (fu <= fx):
                    if (u >= x):
                        a=x
                    else:
                        b=x

                    #shft(v,w,x,u)
                    v = w
                    w = x
                    x = u
                    #shft(fv,fw,fx,fu)
                    fv = fw
                    fw = fx
                    fx = fu


                else:
                    if (u < x):
                        a=u
                    else:
                        b=u

                    if fu <= fw or w == x:
                        v=w
                        fv=fw
                        w=u
                        fw=fu

                    elif fu <= fv or v == x or v == w:
                        v=u
                        fv=fu

            if it == itmax-1:
                print('brent: exceed maximum iterations')

            ans = xmin

            return ans

        @nb.jit(nopython = True)
        def _inner_loop_for_assigned_(assigned_indexes, _EV_, _vc_an_, _vcn_, _vc_util_, _is_c_, _is_o_):


            ibegin = assigned_indexes[0]
            iend = assigned_indexes[1]

            ind = 0
            for ipar_loop in range(ibegin, iend):

                #we should replace unravel_index_nb with something unflexible one.
                istate, ia = unravel_ip(ipar_loop)

                if _is_c_:
                    an_sup = min(c_supan[ia, is_to_ieps[istate], _is_o_] - 1.e-6, agrid[-1]) #no extrapolation for aprime
                    an_min = amin
                else:#if S
                    an_sup = min(s_supan[ia, iz, _is_o_] - 1.e-6, agrid[-1]) #no extrapolation for aprime
                    an_min = amin
                    #ks = (z**(1./(1.-alpha)))*xi8
                    #an_min = chi*ks #chi*ks

                ans =  _optimize_given_state_(an_min, an_sup, _EV_, ia, istate, _is_c_, _is_o_)
                

                _vc_an_[ind] = ans
                _vcn_[ind] = -_obj_loop_(ans, _EV_, ia, istate, _is_c_, _is_o_)

                if _is_c_:
                    _vc_util_[ind] = get_cstatic(np.array([agrid[ia], ans, epsgrid[is_to_ieps[istate]], _is_o_]))[0]
                else:
                    _vc_util_[ind] = get_sstatic(np.array([agrid[ia], ans, zgrid[is_to_iz[istate]], _is_o_]))[0]
                    
                ind = ind + 1

        @nb.jit(nopython = True)
        def _howard_iteration_(_vmax_y_, _vmax_o_, _bEV_yc_, _bEV_oc_, _bEV_ys_, _bEV_os_, #these vars are just data containers
                               _vn_yc_, _vn_oc_, _vn_ys_, _vn_os_, #these value functions will be updated
                               _v_yc_util_, _v_oc_util_, _v_ys_util_, _v_os_util_,
                               _v_yc_an_, _v_oc_an_, _v_ys_an_, _v_os_an_,               
                               _howard_iter_):

            for it_ho in range(_howard_iter_):
                _vmax_y_[:] = np.fmax(_vn_yc_, _vn_ys_)
                _vmax_o_[:] = np.fmax(_vn_oc_, _vn_os_)                


                # bEV_yc[:] = prob_yo[0,0]*bh*((v_y_max**(1. - mu))@(prob.T)).reshape((1, num_a, num_s)) +\
                #             prob_yo[0,1]*bh*((v_o_max**(1. - mu))@(prob.T)).reshape((1, num_a, num_s))
                # bEV_oc[:] = prob_yo[1,1]*bh*((v_o_max**(1. - mu))@(prob.T)).reshape((1, num_a, num_s)) +\
                #             prob_yo[1,0]*iota*bh*((v_y_max**(1. - mu))@(prob_st)).reshape((1, num_a, 1))
                

                #numba does not support matmul for 3D or higher matrice
                for ia in range(num_a):                
                    _bEV_yc_[0,ia,:] = prob_yo[0,0]*bh*((_vmax_y_[ia,:]**(1. - mu))@(prob.T)).reshape((1, 1, num_s)) +\
                                       prob_yo[0,1]*bh*((_vmax_o_[ia,:]**(1. - mu))@(prob.T)).reshape((1, 1, num_s))

                for ia in range(num_a):
                    _bEV_oc_[0,ia,:] = prob_yo[1,1]*bh*((_vmax_o_[ia,:]**(1. - mu))@(prob.T)).reshape((1, 1, num_s)) +\
                                       prob_yo[1,0]*iota*bh*((_vmax_y_[ia,:]**(1. - mu))@(prob_st))# .reshape((1, 1, 1))
                #does this reshape work?

                _bEV_ys_[:] = _bEV_yc_[:]
                _bEV_os_[:] = _bEV_oc_[:]                
                

                for istate in range(num_s):

                    for ia in range(num_a):
                        _vn_yc_[ia, istate] = (_v_yc_util_[ia, istate] +
                                               fem_peval(_v_yc_an_[ia, istate], agrid, _bEV_yc_[0,:,istate]) )**(1./(1. - mu))
                        _vn_oc_[ia, istate] = (_v_oc_util_[ia, istate] +
                                               fem_peval(_v_oc_an_[ia, istate], agrid, _bEV_oc_[0,:,istate]) )**(1./(1. - mu))
                        _vn_ys_[ia, istate] = (_v_ys_util_[ia, istate] +
                                               fem_peval(_v_ys_an_[ia, istate], agrid, _bEV_ys_[0,:,istate]) )**(1./(1. - mu))
                        _vn_os_[ia, istate] = (_v_os_util_[ia, istate] +
                                               fem_peval(_v_os_an_[ia, istate], agrid, _bEV_os_[0,:,istate]) )**(1./(1. - mu))
            
                        


        @nb.jit(nopython = True)
        def reshape_to_mat(v, val):
            for i in range(len(val)):
                # istate, ia, ikap = unravel_ip(i)
                # v[ia, ikap, istate] = val[i]
                
                istate, ia = unravel_ip(i)
                v[ia, istate] = val[i]


        #initialize variables for VFI            
        v_yc_an_tmp = np.ones((num_assigned))
        vn_yc_tmp = np.ones((num_assigned))
        v_yc_util_tmp = np.ones((num_assigned))

        v_oc_an_tmp = np.ones((num_assigned))
        vn_oc_tmp = np.ones((num_assigned))
        v_oc_util_tmp = np.ones((num_assigned))
        
        v_ys_an_tmp = np.ones((num_assigned))
        v_ys_kapn_tmp = np.ones((num_assigned))
        vn_ys_tmp = np.ones((num_assigned))
        v_ys_util_tmp = np.ones((num_assigned))

        v_os_an_tmp = np.ones((num_assigned))
        v_os_kapn_tmp = np.ones((num_assigned))
        vn_os_tmp = np.ones((num_assigned))
        v_os_util_tmp = np.ones((num_assigned))
        


        v_yc_an_full = None
        vn_yc_full = None
        v_yc_util_full = None

        if rank == 0:
            v_yc_an_full = np.ones((num_total_state))
            vn_yc_full = np.ones((num_total_state))*(-2.)
            v_yc_util_full = np.ones((num_total_state))

        v_oc_an_full = None
        vn_oc_full = None
        v_oc_util_full = None

        if rank == 0:
            v_oc_an_full = np.ones((num_total_state))
            vn_oc_full = np.ones((num_total_state))*(-2.)
            v_oc_util_full = np.ones((num_total_state))
            


        v_ys_an_full = None
        v_ys_kapn_full = None
        vn_ys_full = None
        v_ys_util_full = None

        if rank == 0:
            v_ys_an_full = np.ones((num_total_state))
            v_ys_kapn_full = np.ones((num_total_state))
            vn_ys_full = np.ones((num_total_state))*(-2.)
            v_ys_util_full = np.ones((num_total_state))


        v_os_an_full = None
        v_os_kapn_full = None
        vn_os_full = None
        v_os_util_full = None

        if rank == 0:
            v_os_an_full = np.ones((num_total_state))
            v_os_kapn_full = np.ones((num_total_state))
            vn_os_full = np.ones((num_total_state))*(-2.)
            v_os_util_full = np.ones((num_total_state))


        v_y_max = np.ones((num_a, num_s))
        v_y_maxn = np.ones((num_a, num_s))*100.0
        v_y_maxm1 = np.ones(v_y_max.shape)

        v_o_max = np.ones((num_a, num_s))
        v_o_maxn = np.ones((num_a, num_s))*100.0
        v_o_maxm1 = np.ones(v_o_max.shape)

        
        bEV_yc = np.ones((1, num_a, num_s))
        bEV_oc = np.ones((1, num_a, num_s))
        bEV_ys = np.ones((1, num_a, num_s))
        bEV_os = np.ones((1, num_a, num_s))        

        v_yc_an = np.zeros((num_a, num_s))
        vn_yc = np.ones((num_a, num_s))*100.0
        v_yc_util = np.ones((num_a, num_s))*100.0

        v_oc_an = np.zeros((num_a, num_s))
        vn_oc = np.ones((num_a, num_s))*100.0
        v_oc_util = np.ones((num_a, num_s))*100.0
        

        v_ys_an = np.zeros((num_a, num_s))
        v_ys_kapn = np.zeros((num_a, num_s))
        vn_ys = np.ones((num_a, num_s))*100.0
        v_ys_util = np.ones((num_a, num_s))*100.0

        v_os_an = np.zeros((num_a, num_s))
        #v_os_kapn does not take into account succession.
        #In the sumulatin part, kap will be replaced by la_tilde kap if it is succeeded
        v_os_kapn = np.zeros((num_a, num_s)) 
        vn_os = np.ones((num_a, num_s))*100.0
        v_os_util = np.ones((num_a, num_s))*100.0
                


        tol = 1.0e-8 #was 1.0e-6
        dist = 10000.0
        dist_sub = 10000.0
        it = 0

        ###record some time###
        t1, t2, t3, t4,tyc1, tyc2, tys1, tys2, toc1, toc2, tos1, tos2 = 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0.        
        # t1, t2, t3, t4,tc1, tc2, ts1, ts2 = 0., 0., 0., 0., 0., 0., 0., 0.,
        if rank == 0:
            t1 = time.time()

        ###main VFI iteration###
        while it < max_iter and dist > tol:

            

            ### calculate EV and Bcast ###
            if rank == 0:
                it = it + 1

                bEV_yc[:] = prob_yo[0,0]*bh*((v_y_max**(1. - mu))@(prob.T)).reshape((1, num_a, num_s)) +\
                            prob_yo[0,1]*bh*((v_o_max**(1. - mu))@(prob.T)).reshape((1, num_a, num_s))
                bEV_oc[:] = prob_yo[1,1]*bh*((v_o_max**(1. - mu))@(prob.T)).reshape((1, num_a, num_s)) +\
                            prob_yo[1,0]*iota*bh*((v_y_max**(1. - mu))@(prob_st)).reshape((1, num_a, 1))

                #under the current structure, the following is true. If not, it should be calculated separetely
                bEV_ys[:] = bEV_yc[:]
                bEV_os[:] = bEV_oc[:]                
                # EV[:] = bh*((vmax**(1. - mu))@(prob.T)).reshape((1,) + vmax.shape)
                # EV[:] = ((vmax)@(prob.T)).reshape((1,) + vmax.shape)                

            comm.Bcast([bEV_yc, MPI.DOUBLE])
            comm.Bcast([bEV_oc, MPI.DOUBLE])
            comm.Bcast([bEV_ys, MPI.DOUBLE])                        
            comm.Bcast([bEV_os, MPI.DOUBLE])            
                
            # comm.Bcast([EV, MPI.DOUBLE])
            ### end calculate EV and Bcast ###            


            ###yc-loop begins####            
            if rank == 0:
                tyc1 = time.time()
            _inner_loop_for_assigned_(assigned_state_range, bEV_yc, v_yc_an_tmp, vn_yc_tmp, v_yc_util_tmp, 1, 0)

            if rank == 0:
                tyc2 = time.time()
                print('time for yc = {:f}'.format(tyc2 - tyc1), end = ', ')
            ###yc-loop ends####

            ###oc-loop begins####            
            if rank == 0:
                toc1 = time.time()
            _inner_loop_for_assigned_(assigned_state_range, bEV_oc, v_oc_an_tmp, vn_oc_tmp, v_oc_util_tmp, 1, 1)

            if rank == 0:
                toc2 = time.time()
                print('time for oc = {:f}'.format(toc2 - toc1), end = ', ')
            ###oc-loop ends####

            ###ys-loop begins####            
            if rank == 0:
                tys1 = time.time()
            _inner_loop_for_assigned_(assigned_state_range, bEV_ys, v_ys_an_tmp, vn_ys_tmp, v_ys_util_tmp, 0, 0)

            if rank == 0:
                tys2 = time.time()
                print('time for ys = {:f}'.format(tys2 - tys1), end = ', ')
            ###ys-loop ends####

            ###os-loop begins####            
            if rank == 0:
                tos1 = time.time()
            _inner_loop_for_assigned_(assigned_state_range, bEV_os, v_os_an_tmp, vn_os_tmp, v_os_util_tmp, 0, 1)

            if rank == 0:
                tos2 = time.time()
                print('time for os = {:f}'.format(tos2 - tos1), end = ', ')
            ###os-loop ends####


            ####policy function iteration starts#####

            comm.Gatherv(vn_yc_tmp,[vn_yc_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            comm.Gatherv(vn_oc_tmp,[vn_oc_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])            

            comm.Gatherv(vn_ys_tmp,[vn_ys_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            comm.Gatherv(vn_os_tmp,[vn_os_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            
            comm.Gatherv(v_yc_an_tmp,[v_yc_an_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            comm.Gatherv(v_oc_an_tmp,[v_oc_an_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            
            comm.Gatherv(v_ys_an_tmp,[v_ys_an_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            comm.Gatherv(v_os_an_tmp,[v_os_an_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])            
            
            comm.Gatherv(v_yc_util_tmp,[v_yc_util_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            comm.Gatherv(v_oc_util_tmp,[v_oc_util_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])            

            comm.Gatherv(v_ys_util_tmp,[v_ys_util_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            comm.Gatherv(v_os_util_tmp,[v_os_util_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])            
            
            
            if rank == 0:

                reshape_to_mat(v_yc_an, v_yc_an_full)
                reshape_to_mat(v_oc_an, v_oc_an_full)
                
                reshape_to_mat(v_ys_an, v_ys_an_full)
                reshape_to_mat(v_os_an, v_os_an_full)
                
                reshape_to_mat(v_yc_util, v_yc_util_full)
                reshape_to_mat(v_oc_util, v_oc_util_full)                
                
                reshape_to_mat(v_ys_util, v_ys_util_full)
                reshape_to_mat(v_os_util, v_os_util_full)
                
                reshape_to_mat(vn_yc, vn_yc_full)
                reshape_to_mat(vn_oc, vn_oc_full)
                
                reshape_to_mat(vn_ys, vn_ys_full)
                reshape_to_mat(vn_os, vn_os_full)                

                ####policy function iteration starts#####
                if max_howard_iter > 0:
                    #print('Starting Howard Iteration...')
                    t3 = time.time()

                    _howard_iteration_(v_y_maxn, v_o_maxn, bEV_yc, bEV_oc, bEV_ys, bEV_os, #these vars are just data containers
                                       vn_yc, vn_oc, vn_ys, vn_os, #these value functions will be updated 
                                       v_yc_util, v_oc_util, v_ys_util, v_os_util,
                                       v_yc_an, v_oc_an, v_ys_an, v_os_an,                               
                                       # v_ys_kapn, v_os_kapn,
                                       max_howard_iter)
                    
                    # _howard_iteration_(vmaxn, vcn, vsn, vc_an, vc_util, vs_an, vs_util, max_howard_iter)


                if max_howard_iter > 0:
                    t4 = time.time()
                    print('time for HI = {:f}'.format(t4 - t3), end = ', ') 

                ####policy function iteration ends#####


            #### post_calc ####
            if rank == 0:

                pol_y = vn_yc > vn_ys
                v_y_maxn[:] = np.fmax(vn_yc, vn_ys)
                pol_o = vn_oc > vn_os
                v_o_maxn[:] = np.fmax(vn_oc, vn_os)

                dist_y_sub = np.max(np.abs(v_y_maxn - v_y_max))
                dist_o_sub = np.max(np.abs(v_o_maxn - v_o_max))
                dist_sub = max(dist_y_sub, dist_o_sub)
                
                dist_y = np.max(np.abs(v_y_maxn - v_y_max) / (1. + np.abs(v_y_maxn)))
                dist_o = np.max(np.abs(v_o_maxn - v_o_max) / (1. + np.abs(v_o_maxn)))
                dist = max(dist_y, dist_o)
                print('')
                print('{}th loop. dist = {:f}, dist_sub = {:f}.'.format(it, dist, dist_sub), end = ',')
                v_y_maxm1[:] = v_y_max[:]
                v_o_maxm1[:] = v_o_max[:]
                v_y_max[:] = v_y_maxn[:]
                v_o_max[:] = v_o_maxn[:]                
                

            it = comm.bcast(it)
            dist = comm.bcast(dist)
            dist_sub = comm.bcast(dist_sub)

            
        ###end main VFI###

        ###post-calc###
        if rank == 0:
            t2 = time.time()
            print('Iteration terminated at {}th loop. dist = {}'.format(it, dist))
            print('Elapsed time: {:f} sec'.format(t2 - t1)) 


        #return value and policy functions

        ###Bcast the result###

        comm.Bcast([v_yc_an, MPI.DOUBLE])
        comm.Bcast([v_oc_an, MPI.DOUBLE])
        comm.Bcast([v_ys_an, MPI.DOUBLE])
        comm.Bcast([v_os_an, MPI.DOUBLE])

        comm.Bcast([vn_ys, MPI.DOUBLE])
        comm.Bcast([vn_os, MPI.DOUBLE])        
        comm.Bcast([vn_yc, MPI.DOUBLE])
        comm.Bcast([vn_oc, MPI.DOUBLE])        


        #return policy function
        self.v_yc_an = v_yc_an
        self.v_oc_an = v_oc_an        
        self.v_ys_an = v_ys_an
        self.v_os_an = v_os_an        
        self.vn_yc = vn_yc
        self.vn_oc = vn_oc        
        self.vn_ys = vn_ys
        self.vn_os = vn_os        

        self.bEV_yc = bEV_yc
        self.bEV_oc = bEV_oc
        self.bEV_ys = bEV_ys 
        self.bEV_os = bEV_os 


#     def generate_shocks(self):

#         """
#         return:
#         data_is_o
#         data_i_s
#         """

#         prob = self.prob
#         prob_yo = self.prob_yo
#         prob_st = self.prob_st
#         num_s = self.num_s

#         #simulation parameters
#         sim_time = self.sim_time
#         num_total_pop = self.num_total_pop
        
        
        
#         ###codes to generate shocks###
#         @nb.jit(nopython = True)
#         def transit(i, r, _prob_):
#             num_s = _prob_.shape[0]

#             if r <= _prob_[i,0]:
#                 return 0

#             for j in range(1, num_s):

#                 #print(np.sum(_prob_[i,0:j]))
#                 if r <= np.sum(_prob_[i,0:j]):
#                     return j - 1

#             if r > np.sum(_prob_[i,0:-1]) and r <= 1.:
#                 return num_s - 1

#             print('error')

#             return -1

#         @nb.jit(nopython = True)
#         def draw(r, _prob_st_):
#             num_s = len(_prob_st_)

#             if r <= _prob_st_[0]:
#                 return 0

#             for j in range(1, num_s):

#                 #print(np.sum(_prob_st_[0:j]))
#                 if r <= np.sum(_prob_st_[0:j]):
#                     return j - 1

#             if r > np.sum(_prob_st_[0:-1]) and r <= 1.:
#                 return num_s - 1

#             print('error')

#             return -1    
        

#         np.random.seed(1) #fix the seed
#         data_rnd_s = np.random.rand(num_total_pop, 2*sim_time+1) 
#         data_rnd_yo = np.random.rand(num_total_pop,2*sim_time+1)
        
#         @nb.jit(nopython = True, parallel = True)
#         #@nb.jit(nopython = True)        
#         def calc_trans(_data_, _rnd_, _prob_):
#             num_entity, num_time = _rnd_.shape

# #            for i in range(num_entity):
#             for i in nb.prange(num_entity):                
#                 for t in range(1, num_time):                    
                    
#                     _data_[i, t] = transit(_data_[i, t-1], _rnd_[i, t], _prob_)

#         print('generating is_o...')
#         data_is_o = np.zeros((num_total_pop, 2*sim_time+1), dtype = int) #we can't set dtype = bool
#         calc_trans(data_is_o, data_rnd_yo, prob_yo)
#         print('done')

#         #transition of s = (eps, z) depends on young-old transition.
#         @nb.jit(nopython = True, parallel = True)
# #        @nb.jit(nopython = True)        
#         def calc_trans_shock(_data_, _data_is_o_, _rnd_s_, _prob_s_, _prob_s_st_):
#             num_entity, num_time = _rnd_s_.shape

#             for i in nb.prange(num_entity):
# #            for i in range(num_entity):
#                 for t in range(1, num_time):
#                     is_o = _data_is_o_[i,t]
#                     is_o_m1 = _data_is_o_[i,t-1]

#                     if is_o_m1 and not (is_o): #if s/he dies and reborns, prod. shocks are drawn from the stationary dist
#                         _data_[i, t] = draw(_rnd_s_[i, t], _prob_s_st_)
                        
#                     else:
#                         _data_[i, t] = transit(_data_[i, t-1], _rnd_s_[i, t], _prob_s_)
#         print('generating i_s...')
#         data_i_s = np.ones((num_total_pop, 2*sim_time+1), dtype = int) * (num_s // 2)
#         calc_trans_shock(data_i_s, data_is_o, data_rnd_s, prob, prob_st)
#         print('done')
        
#         return data_i_s[:, sim_time:2*sim_time], data_is_o[:, sim_time:2*sim_time+1]
        

        
    #def get_obj(w, p, rc, vc_an, vs_an, vs_kapn, vcn, vsn):
    def simulate_model(self):
        for variable in self.__dict__ : exec(variable+'= self.'+variable)                         
                         
        prob = self.prob
        prob_yo = self.prob_yo
        
        #load the value and policy functions

        v_yc_an = self.v_yc_an
        v_oc_an = self.v_oc_an        
        v_ys_an = self.v_ys_an
        v_os_an = self.v_os_an        
        
        vn_yc = self.vn_yc
        vn_oc = self.vn_oc        
        vn_ys = self.vn_ys
        vn_os = self.vn_os

        vn_y = np.fmax(vn_yc, vn_ys)
        vn_o = np.fmax(vn_oc, vn_os)

        bEV_yc = prob_yo[0,0]*bh*((vn_y**(1. - mu))@(prob.T)).reshape((1, num_a, num_s)) +\
                 prob_yo[0,1]*bh*((vn_o**(1. - mu))@(prob.T)).reshape((1, num_a, num_s))
        bEV_oc = prob_yo[1,1]*bh*((vn_o**(1. - mu))@(prob.T)).reshape((1, num_a, num_s)) +\
                 prob_yo[1,0]*iota*bh*((vn_y**(1. - mu))@(prob_st)).reshape((1, num_a, 1))

        #under the current structure, the following is true. If not, it should be calculated separetely
        bEV_ys = bEV_yc.copy()
        bEV_os = bEV_oc.copy()            
        
        

        @nb.jit(nopython = True)
        def unravel_ip(i_aggregated_state):

            istate, ia = unravel_index_nb(i_aggregated_state, num_s, num_a)
            #ia, ikap, istate = unravel_index_nb(i_aggregated_state, num_a, num_kap, num_s)
            return istate, ia

        get_cstatic = self.generate_cstatic()
        get_sstatic = self.generate_sstatic()


        ### start parameters for MPI ###
        m = num_total_pop // size
        r = num_total_pop % size

        assigned_pop_range =  (rank*m+min(rank,r)), ((rank+1)*m+min(rank+1,r))
        num_pop_assigned = assigned_pop_range[1] - assigned_pop_range[0]


        all_assigned_pop_range = np.ones((size, 2))*(-2.)
        all_num_pop_assigned = ()
        all_istart_pop_assigned = ()

        for irank in range(size):

            all_istart_pop_assigned += (int(irank*m+min(irank,r)), )
            all_assigned_pop_range[irank,0] = irank*m+min(irank,r)
            all_assigned_pop_range[irank,1] = (irank+1)*m+min(irank+1,r)
            all_num_pop_assigned += (int(all_assigned_pop_range[irank,1] - all_assigned_pop_range[irank,0]),) 
        ### end parameters for MPI ###    


        #data container for each node
        data_a_elem = np.ones((num_pop_assigned, sim_time))*4.0
        data_i_s_elem = np.ones((num_pop_assigned, sim_time), dtype = int)*7
        data_is_c_elem = np.zeros((num_pop_assigned, sim_time), dtype = bool) 
        data_is_c_elem[0:int(num_pop_assigned*0.7), 0] = True
        data_is_o_elem = np.zeros((num_pop_assigned, sim_time+1), dtype = bool)
        


        

        #main data container
        data_a = None
        data_i_s = None
        data_is_c = None
        data_is_o = None        

        if rank == 0:
            data_a = np.zeros((num_total_pop, sim_time))
            data_i_s = np.zeros((num_total_pop, sim_time), dtype = int)
            data_is_c = np.zeros((num_total_pop, sim_time), dtype = bool)
            data_is_o = np.zeros((num_total_pop, sim_time+1), dtype = bool) 

    #     ###codes to generate shocks###
    #     @nb.jit(nopython = True)
    #     def transit(i, r):

    #         if r <= prob[i,0]:
    #             return 0

    #         for j in range(1, num_s):

    #             #print(np.sum(prob[i,0:j]))
    #             if r <= np.sum(prob[i,0:j]):
    #                 return j - 1

    #         if r > np.sum(prob[i,0:-1]) and r <= 1.:
    #             return num_s - 1

    #         print('error')

    #         return -1    

    #     np.random.seed(rank) #fix the seed
    #     data_rnd = np.random.rand(num_pop_assigned, sim_time)

    #     @nb.jit(nopython = True)
    #     def calc_trans(data_i_s_):
    #         for t in range(1, sim_time):
    #             for i in range(num_pop_assigned):
    #                 data_i_s_[i, t] = transit(data_i_s_[i, t-1], data_rnd[i, t])
    #     calc_trans(data_i_s_elem)
    #     ###end codes to generate shocks###


        ###load shock data### self.path_to_data_i_s
        # data_i_s_import = np.load(Econ.path_to_data_i_s)
        # data_i_s_import = np.load('./input_data/data_i_s.npy')
        # data_i_s_elem[:] = data_i_s_import[assigned_pop_range[0]:assigned_pop_range[1],0:sim_time]


        ###load productivity shock data###

        data_i_s_elem[:] = np.load(self.path_to_data_i_s + '_' + str(rank) + '.npy')
        data_is_o_elem[:] = np.load(self.path_to_data_is_o + '_' + str(rank) + '.npy')
        

       
        # del data_i_s_import

        @nb.jit(nopython = True)
        def calc(data_a_, data_i_s_, data_is_c_, data_is_o_):

            for t in range(1, sim_time):
                for i in range(num_pop_assigned):

                    a = data_a_[i, t-1]

                    is_o = data_is_o_[i,t]                    
                    istate = data_i_s_[i, t]
                    
                    eps = epsgrid[is_to_ieps[istate]]
                    z = zgrid[is_to_iz[istate]]


                    if not is_o: #if young

                        an_c = femeval(a, agrid, v_yc_an[:,istate])
                        an_s = femeval(a, agrid, v_ys_an[:,istate])

                        val_c = (get_cstatic([a, an_c, eps, is_o])[0] + fem_peval(an_c, agrid, bEV_yc[0, :, istate]))**(1./(1.- mu))
                        val_s = (get_sstatic([a, an_s, z  , is_o])[0] + fem_peval(an_s, agrid, bEV_ys[0, :, istate]))**(1./(1.- mu))

                    else:

                        an_c = femeval(a, agrid, v_oc_an[:,istate])
                        an_s = femeval(a, agrid, v_os_an[:,istate])

                        val_c = (get_cstatic([a, an_c, eps, is_o])[0] + fem_peval(an_c, agrid, bEV_oc[0, :, istate]))**(1./(1.- mu))
                        val_s = (get_sstatic([a, an_s, z  , is_o])[0] + fem_peval(an_s, agrid, bEV_os[0, :, istate]))**(1./(1.- mu))
                    
                    if (val_c == val_s):
                        print('error: val_c == val_s')

                    i_c = val_c > val_s

                    an = i_c * an_c + (1. - i_c) * an_s

                    # if (an < chi*xi8) and not i_c:
                    #     print('simulation error: an < k_s but S. t = ', t , ', i = ' , i)
                        
                    if (an < amin):
                        print('simulation error: an < amin . t = ', t , ', i = ' , i)
                        

                    data_a_[i, t] = an
                    data_is_c_[i, t] = i_c

        calc(data_a_elem, data_i_s_elem, data_is_c_elem, data_is_o_elem)

        comm.Gatherv(data_a_elem, [data_a, all_num_pop_assigned, all_istart_pop_assigned,  MPI.DOUBLE.Create_contiguous(sim_time).Commit() ])

        comm.Gatherv(data_i_s_elem, [data_i_s, all_num_pop_assigned, all_istart_pop_assigned,  MPI.LONG.Create_contiguous(sim_time).Commit() ])   
        comm.Gatherv(data_is_c_elem, [data_is_c, all_num_pop_assigned, all_istart_pop_assigned,  MPI.BOOL.Create_contiguous(sim_time).Commit() ])
        comm.Gatherv(data_is_o_elem, [data_is_o, all_num_pop_assigned, all_istart_pop_assigned,  MPI.BOOL.Create_contiguous(sim_time+1).Commit() ])


        #calculate other variables

        data_ss = None

        if rank == 0:

            data_ss = np.ones((num_total_pop, 18)) * (-2.0)

            t = -1
            for i in range(num_total_pop):

                #need to check the consistency within variables... there may be errors...
                if data_is_c[i, t]: 

                    a = data_a[i, t-1]
                    an = data_a[i, t]
                    eps = epsgrid[is_to_ieps[data_i_s[i, t]]]
                    is_o = data_is_o[i, t]

                    data_ss[i,0] = 1.
                    data_ss[i,1] = a
                    data_ss[i,2] = np.nan
                    data_ss[i,3] = an
                    data_ss[i,4] = np.nan
                    data_ss[i,5] = eps
                    data_ss[i,6:16] = get_cstatic(np.array([a, an, eps, is_o]))[1:]
                    data_ss[i,17] = is_o

                    #return u, cc, cs, cagg, l ,n, np.nan, np.nan,   ibracket, taun, psin + transfer, is_o

                else:

                    a = data_a[i, t-1]
                    an = data_a[i, t]
                    z = zgrid[is_to_iz[data_i_s[i, t]]]
                    is_o = data_is_o[i, t]                    

                    data_ss[i,0] = 0.
                    data_ss[i,1] = a
                    data_ss[i,2] = np.nan
                    data_ss[i,3] = an
                    data_ss[i,4] = np.nan
                    data_ss[i,5] = z

                    
                    data_ss[i,6:17] = get_sstatic(np.array([a, an, z, is_o]))[1:]
                    data_ss[i,17] = is_o                    

                    #return u, cc, cs, cagg, lbar, -1.0e20, ks, ys, ibracket, taub[ibracket], psib[ibracket]+transfer, ns, is_o



        self.data_a = data_a
        self.data_i_s = data_i_s
        self.data_is_c = data_is_c
        self.data_is_o = data_is_o        
        self.data_ss = data_ss


        self.calc_moments()

        return

    def calc_age(self):

        #import data from Econ

        #simulation parameters
        sim_time = self.sim_time
        num_total_pop = self.num_total_pop

        #load main simlation result
        data_is_c = self.data_is_c
        data_is_s = ~data_is_c

        data_is_o = self.data_is_o
        data_is_y = ~data_is_o

        data_is_born = np.zeros((num_total_pop, sim_time+1), dtype = bool)
        data_is_born[:,1:] = (~data_is_o[:,1:])*(data_is_o[:,:-1])

        self.data_is_born = data_is_born
        
        s_age = np.ones(num_total_pop, dtype = int) * -1
        c_age = np.ones(num_total_pop, dtype = int) * -1

        sind_age = np.ones(num_total_pop, dtype = int) * -1
        cind_age = np.ones(num_total_pop, dtype = int) * -1
        
        o_age = np.ones(num_total_pop, dtype = int) * -1
        y_age = np.ones(num_total_pop, dtype = int) * -1

        ind_age = np.ones(num_total_pop, dtype = int) * -1
        

        @nb.jit(nopython = True, parallel = True)
        def _calc_age_(_data_is_, _age_):
            for i in nb.prange(num_total_pop):
                if not _data_is_[i,-1]:
                    _age_[i] = -1
                else:
                    t = 0
                    while t < sim_time:
                        if _data_is_[i, -t - 1]:
                            _age_[i] = t
                        else:
                            break
                        t = t + 1


        @nb.jit(nopython = True, parallel = True)
#        @nb.jit(nopython = True)
        def _calc_ind_age_(_data_is_born_, _age_):
            for i in nb.prange(num_total_pop):
                t = 0

                while t < sim_time:
                    if _data_is_born_[i, -t - 1]:
                        _age_[i] = t
                        break
                    else:
                        t = t+1

        @nb.jit(nopython = True, parallel = True)
        def _calc_age_adjust_born_(_data_is_, _data_is_born_, _age_):
            for i in nb.prange(num_total_pop):
                if not _data_is_[i,-1]:
                    _age_[i] = -1
                else:
                    t = 0
                    while t < sim_time:
                        if _data_is_[i, -t-1] and not _data_is_born_[i, -t-1]: #as long as they stay the same occupation and NOT age 0
                            _age_[i] = t
                            
                        elif _data_is_[i, -t-1] and _data_is_born_[i, -t-1]:
                            _age_[i] = t
                            break
                    
                        else:
                            break

                        t = t + 1
                        
                        

        _calc_age_(data_is_c, c_age)                
        _calc_age_(data_is_s, s_age)                                        
        _calc_age_(data_is_y[:,0:-1], y_age)
        _calc_age_(data_is_o[:,0:-1], o_age)
        _calc_age_adjust_born_(data_is_c, data_is_born[:,0:-1], cind_age)
        _calc_age_adjust_born_(data_is_s, data_is_born[:,0:-1], sind_age)        
        _calc_ind_age_(data_is_born[:,0:-1], ind_age)

        
        # for i in range(num_total_pop):
        #     if data_is_c[i,-1]:
        #         s_age[i] = -1
        #     else:
        #         t = 0
        #         while t < sim_time:
        #             if data_is_s[i, -t - 1]:
        #                 s_age[i] = t
        #             else:
        #                 break
        #             t = t + 1


        # for i in range(num_total_pop):
        #     if data_is_s[i,-1]:
        #         c_age[i] = -1
        #     else:
        #         t = 0
        #         while t < sim_time:
                    
        #             if data_is_c[i, -t - 1]:
        #                 c_age[i] = t
        
        #             else:
        #                 break
        #             t = t + 1
        

        self.s_age = s_age
        self.c_age = c_age
        self.sind_age = sind_age
        self.cind_age = cind_age
        self.y_age = y_age
        self.o_age = o_age
        self.ind_age = ind_age        
        
        return
        

    def calc_moments(self):

        for variable in self.__dict__ : exec(variable+'= self.'+variable, locals(), globals())
        Econ = self

        #load main simlation result
        data_a = self.data_a
        # data_kap = self.data_kap
        data_i_s = self.data_i_s
        data_is_c = self.data_is_c
        data_ss = self.data_ss


        #print moments

        mom0 = None
        mom1 = None
        mom2 = None
        mom3 = None
        mom4 = None
        mom5 = None
        mom6 = None
        mom7 = None

        if rank == 0:

            print('max of a in simulation = {}'.format(np.max(data_a)))
            print('min of a in simulation = {}'.format(np.min(data_a)))

            print('')
            print(f'min of agrid = {agrid[0]}')
            print(f'max of agrid = {agrid[-1]}')            
            print('')
            
            print('amax = {}'.format(np.max(data_a)))
            print('amin = {}'.format(np.min(data_a)))
            # print('kapmax = {}'.format(np.max(data_kap)))
            # print('kapmin = {}'.format(np.min(data_kap)))

            t = -1

            # data_ss
            # 0: is_c
            # 1: a
            # 2: kap
            # 3: an
            # 4: kapn
            # 5: eps or z
            # 6: cc
            # 7: cs
            # 8: cagg
            # 9: l or lbar
            # 10: n or NAN
            # 11: NAN or ks
            # 12: NAN or ys
            # 13: i_bracket
            # 14: taun[] or taub[]
            # 15: psin[] or psib[] + trans_retire
            # 16: ns
            # 17: is_o

            
            EIc = np.mean(data_ss[:,0])
            Ea = np.mean(data_ss[:,1])
            Ekap = np.mean(data_ss[:,2])
            Ecc = np.mean(data_ss[:,6])
            Ecs = np.mean(data_ss[:,7])
            El = np.mean(data_ss[:,9])
            En = np.mean(data_ss[:,5]* data_ss[:,10] * (data_ss[:,0])) #efficient labor
            
            

            Eks = np.mean(data_ss[:,11] * (1. - data_ss[:,0]))
            Eys = np.mean(data_ss[:,12] * (1. - data_ss[:,0]))


            Ex = np.nan
            Ehkap = np.nan
            Ehy = np.nan
            Eh = np.nan
            Ens = np.mean(data_ss[:,16] * (1. - data_ss[:,0]))
                         
            # Ex = np.mean(data_ss[:,13] * (1. - data_ss[:,0]))                         
            # Ehkap = np.mean(data_ss[:,11] * (1. - data_ss[:,0]))
            # Ehy = np.mean(data_ss[:,10] * (1. - data_ss[:,0]))
            # Eh = np.mean(data_ss[:,12] * (1. - data_ss[:,0]))            
            # Ens = np.mean(data_ss[:,16] * (1. - data_ss[:,0])) #new! labor supply for each firms


            Ecagg_c = np.mean((data_ss[:,6] + p*data_ss[:,7] )*data_ss[:,0])
            Ecagg_s = np.mean((data_ss[:,6] + p*data_ss[:,7] )*(1. - data_ss[:,0]))

            wepsn_i = w*data_ss[:,5]*data_ss[:,10]*data_ss[:,0]
            ETn = np.mean((data_ss[:,14]*wepsn_i - data_ss[:,15])*data_ss[:,0])
            #transfer subtracted      

            bizinc_i = (p*data_ss[:,12] - (rs + delk)*data_ss[:,11] - w*data_ss[:,16])*(1.-data_ss[:,0])
            ETm = np.mean((data_ss[:,14]*bizinc_i - data_ss[:,15])*(1.-data_ss[:,0])) #transfer subtracted

            E_transfer = np.mean(data_ss[:,15]) #includes
            ETr = np.mean(trans_retire*data_ss[:,17])            


            nc = En - Ens
            
            self.nc = nc
            self.En = En
            self.Ens = Ens

            kc = nc*kcnc_ratio
            
            # kc = ((w/(1. - theta)/A)**(1./theta))*nc
            

            yc = A * (kc**theta)*(nc**(1.-theta))
            yc_sub = Ecc  + Ex+ (grate + delk)*(kc + Eks) + g + xnb - yn

            Tc = tauc*(Ecc + p*Ecs)
            Tp = taup*(yc - w*nc - delk*kc)
            Td = taud*(yc - w*nc - (grate + delk)*kc - Tp)

            #b = (Tc + ETn + ETm + Tp + Td - g)/(rbar - grate) #old def
            b = Ea - (1. - taud)*kc - Eks
    #         netb = (grate + delk)*b ##typo
            netb = (rbar - grate)*b
            tax_rev = Tc + ETn + ETm + Td + Tp + E_transfer #only tau*() part,

            GDP = yc + yn + p*Eys
            C = Ecc + p*Ecs
            xc = (grate + delk)*kc
            Exs = (grate + delk)*Eks

            def gini(array):
                """Calculate the Gini coefficient of a numpy array."""
                # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
                # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
                array = array.flatten() #all values are treated equally, arrays must be 1d
                if np.amin(array) < 0:
                    array -= np.amin(array) #values cannot be negative
                array += 0.0000001 #values cannot be 0
                array = np.sort(array) #values must be sorted
                index = np.arange(1,array.shape[0]+1) #index per array element
                n = array.shape[0]#number of array elements
                return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient




    #         print()
    #         print('yc = {}'.format(yc)) 
    #         print('nc = {}'.format(nc))
    #         print('kc = {}'.format(kc))
    #         print('Tc = {}'.format(Tc))
    #         print('Tp = {}'.format(Tp))
    #         print('Td = {}'.format(Td))
    #         print('b = {}'.format(b))

            print('')
            print('RESULT')
            print('Simulation Parameters')
            print('Simulation Periods = ', sim_time)
            print('Simulation Pops = ', num_total_pop)
            print('')
            print('Prices')

            print('Wage (w) = {}'.format(w))
            print('S-good price (p) = {}'.format(p))
            print('Interest rate (r_c) = {}'.format(rc))

            #instead of imposing labor market clearing condition
            # mom0 = 1. - (1. - theta)*yc/(w*nc)
            mom0 = 1. - Ecs/Eys
            mom1 = 1. - (Ecc  + (grate + delk)*(kc + Eks) + g + xnb - yn)/yc
            mom2 = 1. - (tax_rev - E_transfer - netb)/g            
            print('')

            # print('1-(1-thet)*yc/(E[w*eps*n]) = {}'.format(mom0))
            print('1-E(cs)/E(ys) = {}'.format(mom0))
            print('1-(Ecc+(grate+delk)*(kc + Eks)+ g + xnb - yn)/yc = {}'.format(mom1))            
            #print('1-((1-taud)kc+E(ks)+b)/Ea = {}'.format(1. - (b + (1.- taud)*kc + Eks)/Ea))
            print('1-(tax-tran-netb)/g = {}'.format(mom2))


            print('')
            print('Important Moments')
            print('  Financial assets(Ea) = {}'.format(Ea))
            print('  Sweat assets(Ekap) = {}'.format(Ekap))
            print('  Govt. debt(b) = {}'.format(b))
            print('  S-Corp Rental Capital (ks) =  {}'.format(Eks))
            print('  Ratio of C-corp workers(EIc) = {}'.format(EIc))
            print('  GDP(yc + yn + p*Eys) = {}'.format(GDP))

            print('')
            print('C-corporation production:')
            print('  Consumption(Ecc) = {}'.format(Ecc))
            print('  Investment((grate+delk)*kc) = {}'.format(kc*(grate + delk)))
            print('  Govt purchase(g) = {}'.format(g))
            print('  Output(yc) = {}'.format(yc))
            print('  Capital(kc) = {}'.format(kc))
            print('  Eff. Worked Hours Demand(nc) = {}'.format(nc)) 

            print('')
            print('S-corporation production:')
            print('  Consumption(Ecs) = {}'.format(Ecs))
            print('  Output(Eys) = {}'.format(Eys))
            print('    Output in prices(p*Eys) = {}'.format(p*Eys))            
            print('  investment, sweat(Ex (or Ee)) = {}'.format(Ex))
            print('  Hours, production(Ehy) = {}'.format(Ehy))
            print('  Hours, sweat(Ehkap) = {}'.format(Ehkap))
            print('  Sweat equity (kap) = {}'.format(Ekap))
            print('  Eff. Worked Hours Demand(Ens) = {}'.format(Ens))             

            print('')
            print('National Income Shares (./GDP):')
            print('  C-wages (w*nc) = {}'.format((w*nc)/GDP))
            print('  Rents(rc*kc + rs*Eks) = {}'.format((rc*kc + rs*Eks)/GDP))
            print('  Sweat(p*Eys - (rs+delk)*Eks)) = {}'.format((p*Eys - (rs+delk)*Eks)/GDP))
            print('      Pure Sweat(p*Eys - (rs+delk)*Eks - w*Ens)) = {}'.format((p*Eys - (rs+delk)*Eks - w*Ens)/GDP))
            print('      S-wage(w*Ens)) = {}'.format((w*Ens)/GDP))                        
            print('  Deprec.(delk*(kc+Eks)) = {}'.format((delk*(kc+Eks))/GDP))
            print('  NonBusiness income(yn) = {}'.format(yn/GDP))
            print('    Sum = {}'.format((w*nc + rc*kc + rs*Eks+ p*Eys - (rs+delk)*Eks + delk*(kc+Eks) + yn)/GDP))

            self.puresweat = (p*Eys - (rs+delk)*Eks - w*Ens)/GDP

            
            print('')
            print('National Product Shares (./GDP):')
            print('  Consumption(C) = {}'.format(C/GDP))
            print('  physical Investments(xc) = {}'.format((xc)/GDP))
            print('  physical Investments(Exs) = {}'.format((Exs)/GDP))
            print('  Govt Consumption(g) = {}'.format(g/GDP))
            print('  Nonbusiness investment(xnb) = {}'.format((xnb)/GDP))
            print('  sweat    Investment(Ex) = {}'.format(Ex))
            print('    Sum = {}'.format((C+xc+Exs+g+xnb)/GDP)) #Ex is removed for now

            print('')
            print('Govt Budget:')
            print('  Public Consumption(g) = {}'.format(g))
            print('  Net borrowing(netb) = {}'.format(netb))
            print('  Transfer  = {}'.format(E_transfer))
            print('  Tax revenue(tax_rev) = {}'.format(tax_rev))

            print('')
            print('Gini Coefficients:')
            print('  Financial Assets = {}'.format(gini(data_ss[:,1])))
            print('  Sweats Assets = {}'.format(gini(data_ss[:,2])))
            print('  C-wages (wepsn) (S\' wage is set to zero)= {}'.format(gini(wepsn_i)))
            print('  C-wages (wepsn) (conditional on C)= {}'.format(gini(wepsn_i[data_ss[:,0] == True])))            
            print('  S-inc (pys - (rs +delk) - wns - x) (C\'s is set to zero )= {}'.format(gini(bizinc_i*(1. - data_ss[:,0]))))
            print('  S-inc (pys - (rs +delk) - wns - x) (conditional on S)= {}'.format(gini(bizinc_i[data_ss[:,0] == False])))            
        #     print('  S-income = {}'.format(gini((p*data_ss[:,14]) * (1. - data_ss[:,0]))))
        #     print('  Total Income'.format('?'))
        #     print()


            print('')
            print('Characteristics of Owners:')
            print('  Frac of S-corp owner = {}'.format(1. - EIc))
            print('  Switched from S to C = {}'.format(np.mean((1. - data_is_c[:, t-1]) * (data_is_c[:, t]))))
            print('  Switched from C to S = {}'.format(np.mean(data_is_c[:, t-1] * (1. - data_is_c[:, t]) )))


            print('')
            print('Acquired N years ago:')

            for t in range(40):
                print(' N is ', t, ', with = {:f}'.format((np.mean(np.all(data_is_c[:,-(t+2):-1] == False, axis = 1)) - np.mean(np.all(data_is_c[:,-(t+3):-1] == False, axis = 1)) )/ np.mean(1. - data_ss[:,0]))) 



            t = -1
            
            print('')
            print('Labor Market')
            print('  Labor Supply(En)   = {}'.format(En))
            print('  Labor Demand of C(nc) = {}'.format(nc))
            print('  Labor Demand of S(Ens) = {}'.format(Ens))
            print('')

            print('')
            print('Additional Moments for the Lifecycle version model')
            print('  Frac of Old         = {}'.format(np.mean(data_ss[:,17])))
            print('  Frac of Old (check) = {}'.format(np.mean(data_is_o[:,t])))
            #add double check
            print('  Frac of Young who was Young, Old = {}, {} '.format(np.mean( (1.-data_is_o[:,t])*(1.-data_is_o[:,t-1])),
                                                                        np.mean( (1.-data_is_o[:,t])*data_is_o[:,t-1])))
            print('  Frac of Old   who was Young, Old = {}, {} '.format(np.mean( (data_is_o[:,t])*(1.-data_is_o[:,t-1])),
                                                                        np.mean( (data_is_o[:,t])*data_is_o[:,t-1])))
            print('  Frac of Young who is  C  ,   S   = {}, {} '.format(np.mean( (1.-data_is_o[:,t])*(data_is_c[:,t])),
                                                                        np.mean( (1.-data_is_o[:,t])*(1.-data_is_c[:,t]))))
            print('  Frac of Old   who is  C  ,   S   = {}, {} '.format(np.mean( (data_is_o[:,t])*(data_is_c[:,t])),
                                                                        np.mean( (data_is_o[:,t])*(1.-data_is_c[:,t]))))

            print('')
            print('')
            # print('  Deterioraton of kapn due to succeession = {}'.format(np.mean(data_kap0[:,t]) - np.mean(data_kap[:,t])))
            # print('  Deterioraton of kap  due to succeession = {}'.format(np.mean(data_kap0[:,t-1]) - np.mean(data_kap[:,t-1])))

            print('  Transfer                = {}'.format(E_transfer))
            print('    Transfer (Non-retire) = {}'.format(E_transfer - ETr))
            print('    Transfer (retire)     = {}'.format(ETr))
            

            print('')
            print('Additional Moments')
            print('  E(phi p ys - x)       = {}'.format(np.nan))
            print('  E(phi p ys - x)/GDP   = {}'.format((np.nan)))
            print('  E(ks)                 = {}'.format(Eks))
            print('  E(ks)/GDP             = {}'.format(Eks/GDP))
            print('  E(nu p ys - w ns)     = {}'.format((nu*p*Eys - w*Ens)))                        
            print('  E(nu p ys - w ns)/GDP = {}'.format((nu*p*Eys - w*Ens)/GDP))            
            
            
            # if self.data_val_sweat is not None:
            #     EVb_sdicount = np.mean(data_val_sweat[:,-1])
            #     print('  EVb (bh un_c/u_c)              = {}'.format(EVb_sdicount))
            #     print('  EVb/GDP (bh un_c/u_c)          = {}'.format(EVb_sdicount/GDP))
                
            # if self.data_val_sweat_1gR is not None:
            #     EVb_1gR = np.mean(data_val_sweat[:,-1])
            #     print('  EVb  ((1+grate)/(1+rbar))      = {}'.format(EVb_1gR))
            #     print('  EVb/GDP ((1+grate)/(1+rbar))   = {}'.format(EVb_1gR/GDP))                            
                

            
            # mom0 = 1. - Ecs/Eys
            # mom1 = 1. - (Ecc  + (grate + delk)*(kc + Eks) + g + xnb - yn)/yc
            # mom2 = 1. - (tax_rev - E_transfer - netb)/g            

            mom3 = (p*Eys - (rs+delk)*Eks)/GDP
            mom4 = Ens/En
            mom5 = (p*Eys - (rs+delk)*Eks - w*Ens)/GDP
            mom6 = nc
            mom7 = 1. - EIc
            
        mom0 = comm.bcast(mom0) #1. - Ecs/Eys
        mom1 = comm.bcast(mom1) # 1. - (Ecc  + Ex+ (grate + delk)*(kc + Eks) + g + xnb - yn)/yc
        mom2 = comm.bcast(mom2) # 1. - (tax_rev - tran - netb)/g
        mom3 = comm.bcast(mom3) # (p*Eys - (rs+delk)*Eks)/GDP)
        mom4 = comm.bcast(mom4) # Ens/En
        mom5 = comm.bcast(mom5) # (p*Eys - (rs+delk)*Eks - w*Ens)/GDP
        mom6 = comm.bcast(mom6) # nc
        mom7 = comm.bcast(mom7) # 1. - EIc
        

        self.moms = [mom0, mom1, mom2, mom3, mom4, mom5, mom6, mom7]

        return
        
        
    def simulate_other_vars(self):

        #load variables
        alpha = self.alpha
        beta = self.beta
        chi = self.chi
        delk = self.delk
        delkap = self.delkap
        eta = self.eta
        g = self.g
        grate = self.grate
        la = self.la
        mu = self.mu
        ome = self.ome
        phi = self.phi
        rho = self.rho
        tauc = self.tauc
        taud = self.taud
        taup = self.taup
        theta = self.theta
        veps = self.veps
        vthet = self.vthet
        xnb = self.xnb
        yn = self.yn
        zeta = self.zeta
        lbar = self.lbar

        agrid = self.agrid
        epsgrid = self.epsgrid
        zgrid = self.zgrid

        prob = self.prob

        is_to_iz = self.is_to_iz
        is_to_ieps = self.is_to_ieps

        amin = self.amin

        num_a = self.num_a
        num_eps = self.num_eps
        num_z = self.num_z
        num_s = self.num_s

        nu = self.nu
        bh = self.bh
        varrho = self.varrho

        w = self.w
        p = self.p
        rc = self.rc

        rbar = self.rbar
        rs = self.rs

        xi1 = self.xi1
        xi2 = self.xi2
        xi3 = self.xi3
        xi4 = self.xi4
        xi5 = self.xi5
        xi6 = self.xi6
        xi7 = self.xi7
        xi8 = self.xi8


        data_a = self.data_a
        data_i_s = self.data_i_s
        data_is_c = self.data_is_c
        data_is_o = self.data_is_o                

        #simulation parameters
        sim_time = self.sim_time
        num_total_pop = self.num_total_pop

        get_cstatic = self.generate_cstatic()
        get_sstatic = self.generate_sstatic()


        @nb.jit(nopython = True, parallel = True)
        def calc_all(data_a_, data_i_s_, data_is_c_, data_is_o_,
                     data_u_, data_cc_, data_cs_, data_cagg_, data_l_, data_n_,  data_ks_, data_ys_,
                     data_i_tax_bracket_, data_tau_, data_psi_, data_ns_):

            for i in nb.prange(num_total_pop):
                for t in range(1, sim_time):

                    istate = data_i_s_[i, t]
                    eps = epsgrid[is_to_ieps[istate]]
                    z = zgrid[is_to_iz[istate]]

                    u = np.nan
                    cc = np.nan
                    cs = np.nan
                    cagg = np.nan
                    l = np.nan
                    n = np.nan
                    
                    hy = np.nan
                    hkap = np.nan
                    h = np.nan
                    ks = np.nan
                    ys = np.nan
                    ns = np.nan

                    ibra = -1000

                    a = data_a_[i, t-1]
                    # kap = data_kap_[i, t-1]

                    an = data_a_[i, t]
                    # kapn = data_kap_[i, t]

                    is_c = data_is_c_[i, t]
                    is_o = data_is_o_[i, t]                    
                    
                    # data_ss
                    # 0: is_c
                    # 1: a
                    # 2: skip
                    # 3: an
                    # 4: skip
                    # 5: eps or z
                    # 6: cc
                    # 7: cs
                    # 8: cagg
                    # 9: l or lbar
                    # 10: n or NAN
                    # 11: NAN or ks
                    # 12: NAN or ys
                    # 13: i_bracket
                    # 14: taun[] or taub[]
                    # 15: psin[] or psib[]
                    # 16: ns
                    # 17: 


                    if is_c:
                        u, cc, cs, cagg, l , n, tmp1, tmp2, ibra, tau, psi = get_cstatic([a, an, eps, is_o])
                    else:
                        u, cc, cs, cagg, l, tmp1, ks, ys, ibra, tau, psi, ns = get_sstatic([a, an, z, is_o])

                    data_u_[i, t] = u
                    data_cc_[i, t] = cc
                    data_cs_[i, t] = cs
                    data_cagg_[i, t] = cagg
                    data_l_[i, t] = l
                    data_n_[i, t] = n
                    # data_mx_[i, t] = mx
                    # data_my_[i, t] = my
                    # data_x_[i, t] = x
                    data_ks_[i, t] = ks
                    data_ys_[i, t] = ys
                    data_i_tax_bracket_[i, t] = ibra

                    data_tau_[i, t] = tau
                    data_psi_[i, t] = psi #includes trans_
                    data_ns_[i, t] = ns                    
                    
        data_u = np.zeros(data_a.shape)
        data_cc = np.zeros(data_a.shape)
        data_cs = np.zeros(data_a.shape)
        data_cagg = np.zeros(data_a.shape)
        data_l = np.zeros(data_a.shape)
        data_n = np.zeros(data_a.shape)
        
        # data_mx = np.zeros(data_a.shape)
        # data_my = np.zeros(data_a.shape)
        # data_x = np.zeros(data_a.shape)
        
        data_ks = np.zeros(data_a.shape)
        data_ys = np.zeros(data_a.shape)
                         
        data_i_tax_bracket = np.zeros(data_a.shape)
        data_tau = np.zeros(data_a.shape)
        data_psi = np.zeros(data_a.shape)
        data_ns = np.zeros(data_a.shape)                                 
                         
        #note that this does not store some impolied values,,,, say div or value of sweat equity
        calc_all(data_a, data_i_s, data_is_c, data_is_o,  ##input
                 data_u, data_cc, data_cs, data_cagg, data_l, data_n, data_ks, data_ys, ##output
                 data_i_tax_bracket, data_tau, data_psi, data_ns)           


        self.data_u = data_u
        self.data_cc = data_cc
        self.data_cs = data_cs
        self.data_cagg = data_cagg
        self.data_l = data_l
        self.data_n = data_n
        # self.data_mx = data_mx
        # self.data_my = data_my
        # self.data_x = data_x
        self.data_ks = data_ks
        self.data_ys = data_ys

        self.data_i_tax_bracket = data_i_tax_bracket 
        self.data_tau = data_tau 
        self.data_psi = data_psi
        self.data_ns = data_ns        
                         

        # self.data_div_sweat = data_div_sweat
        # self.data_val_sweat = data_val_sweat        
        # self.data_val_sweat_bh = data_val_sweat_bh
        # self.data_val_sweat_1gR = data_val_sweat_1gR

        return
    
    def save_result(self, dir_path_save = './save_data/'):
        if rank == 0:
            print('Saving results under ', dir_path_save, '...')


            np.save(dir_path_save + 'agrid', self.agrid)
            # np.save(dir_path_save + 'kapgrid', self.kapgrid)
            np.save(dir_path_save + 'zgrid', self.zgrid)
            np.save(dir_path_save + 'epsgrid', self.epsgrid)                         


            np.save(dir_path_save + 'prob', self.prob)
            np.save(dir_path_save + 'prob_yo', self.prob_yo)            
            np.save(dir_path_save + 'is_to_iz', self.is_to_iz)
            np.save(dir_path_save + 'is_to_ieps', self.is_to_ieps)
            np.save(dir_path_save + 'data_is_o', self.data_is_o[:, -100-1:])


            np.save(dir_path_save + 'taub', self.taub)
            np.save(dir_path_save + 'psib', self.psib)
            np.save(dir_path_save + 'bbracket', self.bbracket)                                                             
            np.save(dir_path_save + 'taun', self.taun)
            np.save(dir_path_save + 'psin', self.psin)
            np.save(dir_path_save + 'nbracket', self.nbracket)                                       
            np.save(dir_path_save + 'data_a', self.data_a[:, -100:])
            # np.save(dir_path_save + 'data_kap', self.data_kap[:, -100:])
            np.save(dir_path_save + 'data_i_s', self.data_i_s[:, -100:])
            np.save(dir_path_save + 'data_is_c', self.data_is_c[:, -100:])
            np.save(dir_path_save + 'data_u', self.data_u[:, -100:])
            np.save(dir_path_save + 'data_cc', self.data_cc[:, -100:])
            np.save(dir_path_save + 'data_cs', self.data_cs[:, -100:])
            np.save(dir_path_save + 'data_cagg', self.data_cagg[:, -100:])
            np.save(dir_path_save + 'data_l', self.data_l[:, -100:])
            np.save(dir_path_save + 'data_n', self.data_n[:, -100:])
            np.save(dir_path_save + 'data_ns', self.data_ns[:, -100:])            
            # np.save(dir_path_save + 'data_mx', self.data_mx[:, -100:])
            # np.save(dir_path_save + 'data_my', self.data_my[:, -100:])
            # np.save(dir_path_save + 'data_x', self.data_x[:, -100:])
            np.save(dir_path_save + 'data_ks', self.data_ks[:, -100:])
            np.save(dir_path_save + 'data_ys', self.data_ys[:, -100:])
            np.save(dir_path_save + 'data_i_tax_bracket', self.data_i_tax_bracket[:, -100:])
            np.save(dir_path_save + 'data_tau', self.data_tau[:, -100:])
            np.save(dir_path_save + 'data_psi', self.data_psi[:, -100:])                                                  
            np.save(dir_path_save + 'data_ss', self.data_ss)


            np.save(dir_path_save + 'v_yc_an', self.v_yc_an)
            np.save(dir_path_save + 'v_oc_an', self.v_oc_an)            
            np.save(dir_path_save + 'v_ys_an', self.v_ys_an)
            np.save(dir_path_save + 'v_os_an', self.v_os_an)
            np.save(dir_path_save + 'vn_yc', self.vn_yc)
            np.save(dir_path_save + 'vn_oc', self.vn_oc)            
            np.save(dir_path_save + 'vn_ys', self.vn_ys)
            np.save(dir_path_save + 'vn_os', self.vn_os)            


            # np.save(dir_path_save + 'sweat_div', self.sweat_div)
            # np.save(dir_path_save + 'sweat_val', self.sweat_val)
            # np.save(dir_path_save + 'sweat_val_bh', self.sweat_val_bh)            
            # np.save(dir_path_save + 'sweat_val_1gR', self.sweat_val_1gR)
            

            # np.save(dir_path_save + 'data_div_sweat', self.data_div_sweat[:, -100:])
            # np.save(dir_path_save + 'data_val_sweat', self.data_val_sweat[:, -100:])
            # np.save(dir_path_save + 'data_val_sweat_bh', self.data_val_sweat_bh[:, -100:])
            # np.save(dir_path_save + 'data_val_sweat_1gR', self.data_val_sweat_1gR[:, -100:])

            np.save(dir_path_save + 'sind_age', self.sind_age)
            np.save(dir_path_save + 'cind_age', self.cind_age)
            np.save(dir_path_save + 's_age', self.s_age)
            np.save(dir_path_save + 'c_age', self.c_age)
            np.save(dir_path_save + 'y_age', self.y_age)
            np.save(dir_path_save + 'o_age', self.o_age)
            np.save(dir_path_save + 'ind_age', self.ind_age)                        
            

            """
            np.save(dir_path_save + 'data_a', self.data_a)
            np.save(dir_path_save + 'data_kap', self.data_kap)
            np.save(dir_path_save + 'data_i_s', self.data_i_s)
            np.save(dir_path_save + 'data_is_c', self.data_is_c)
            np.save(dir_path_save + 'data_u', self.data_u)
            np.save(dir_path_save + 'data_cc', self.data_cc)
            np.save(dir_path_save + 'data_cs', self.data_cs)
            np.save(dir_path_save + 'data_cagg', self.data_cagg)
            np.save(dir_path_save + 'data_l', self.data_l)
            np.save(dir_path_save + 'data_n', self.data_n)
            np.save(dir_path_save + 'data_mx', self.data_mx)
            np.save(dir_path_save + 'data_my', self.data_my)
            np.save(dir_path_save + 'data_x', self.data_x)
            np.save(dir_path_save + 'data_ks', self.data_ks)
            np.save(dir_path_save + 'data_ys', self.data_ys)
            np.save(dir_path_save + 'data_ss', self.data_ss)

            np.save(dir_path_save + 'vc_an', self.vc_an)
            np.save(dir_path_save + 'vs_an', self.vs_an)
            np.save(dir_path_save + 'vs_kapn', self.vs_kapn)
            np.save(dir_path_save + 'vcn', self.vcn)
            np.save(dir_path_save + 'vsn', self.vsn)

            np.save(dir_path_save + 'sweat_div', self.sweat_div)
            np.save(dir_path_save + 'sweat_val', self.sweat_val)

            np.save(dir_path_save + 'data_div_sweat', self.data_div_sweat)
            np.save(dir_path_save + 'data_val_sweat', self.data_val_sweat)

            """

import pickle        
import os.path
from sys import platform

def import_econ(name = 'econ.pickle'):

    if platform == 'darwin':
        from MacOSFile import pickle_load
        return pickle_load(name)
    else:
        with open(name, mode='rb') as f: econ = pickle.load(f)
        return econ


def export_econ(econ, name = 'econ.pickle'):


    if rank == 0:
        if platform == 'darwin':
            from MacOSFile import pickle_dump
            pickle_dump(econ, name)
        else:
            with open(name, mode='wb') as f: pickle.dump(econ, f)   

    return

def split_shock(path_to_data_shock, num_total_pop, size):


    m = num_total_pop // size
    r = num_total_pop % size

    data_shock = np.load(path_to_data_shock + '.npy')
    

    for rank in range(size):
        assigned_pop_range =  (rank*m+min(rank,r)), ((rank+1)*m+min(rank+1,r))
        np.save(path_to_data_shock + '_' + str(rank) + '.npy', data_shock[assigned_pop_range[0]:assigned_pop_range[1], :])

    return



if __name__ == '__main__':
    econ = import_econ()

    econ.get_policy()
    if rank == 0:
        econ.print_parameters()
    econ.simulate_model()

    export_econ(econ)
