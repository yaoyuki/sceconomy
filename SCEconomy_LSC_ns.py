
#import Yuki's library in the directory ./library
import sys
sys.path.insert(0, './library/')


import numpy as np
import numba as nb
###usage
###@nb.jit(nopython = True)


#my library
#import
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
                 chi = None,
                 delk = None,
                 delkap = None,
                 eta = None,
                 g = None,
                 grate = None,
                 la = None,
                 mu = None,
                 ome = None,
                 phi = None,
                 rho = None,
                 tauc = None,
                 taud = None,
                 taum = None,
                 taun = None,
                 taup = None,
                 theta = None,
                 tran = None,
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
                 is_to_iz = None,
                 is_to_ieps = None,
                 amin = None,
                 sim_time = None,
                 num_total_pop = None,
                 A = None,
                 path_to_data_i_s = None):
        

        
        
        self.__set_default_parameters__()
        
        #set the parameters if designated
        #I don't know how to automate these lines
        if alpha is not None: self.alpha = alpha
        if nu is not None: self.nu = nu        
        if beta is not None: self.beta = beta
        if chi is not None: self.chi = chi
        if delk is not None: self.delk = delk    
        if delkap is not None: self.delkap = delkap
        if eta is not None: self.eta = eta
        if g is not None: self.g = g
        if grate is not None: self.grate = grate 
        if la is not None: self.la = la
        if mu is not None: self.mu = mu 
        if ome is not None: self.ome = ome 
        if phi is not None: self.phi = phi
        if rho is not None: self.rho = rho
        if tauc is not None: self.tauc = tauc
        if taud is not None: self.taud = taud
        if taum is not None: self.taum = taum
        if taun is not None: self.taun = taun
        if taup is not None: self.taup = taup
        if theta is not None: self.theta = theta
        if tran is not None: self.tran = tran
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
        if is_to_iz is not None: self.is_to_iz = is_to_iz
        if is_to_ieps is not None: self.is_to_ieps = is_to_ieps
        if amin is not None: self.amin = amin
        if sim_time is not None: self.sim_time = sim_time
        if num_total_pop is not None: self.num_total_pop = num_total_pop
        if A is not None: self.A = A
        if path_to_data_i_s is not None: self.path_to_data_i_s = path_to_data_i_s

        self.__set_implied_parameters__()
    
    def __set_default_parameters__(self):
        
        """
        Load the baseline value
        """
        
        self.__is_price_set__ = False
        self.alpha    = 0.4
        self.nu       = 0.55 #too big?
        self.beta     = 0.98
        self.chi      = 0.0 #param for borrowing constarint
        self.delk     = 0.05
        self.delkap   = 0.05 
        self.eta      = 0.42
        self.g        = 0.234 #govt spending
        self.grate    = 0.02 #gamma, growth rate for detrending
        self.la       = 0.5 #lambda
        self.mu       = 1.5 
        self.ome      = 0.6 #omega
        self.phi      = 0.15 
        self.rho      = 0.5
        self.tauc     = 0.06
        self.taud     = 0.14
        self.taum     = 0.20
        self.taun     = 0.40
        self.taup     = 0.30
        self.theta    = 0.41
        self.tran     = 0.15 #psi
        self.veps     = 0.4
        self.vthet    = 0.4
        self.xnb      = 0.185
        self.yn       = 0.451
        self.zeta     = 1.0 #totally tentative
        self.lbar     = 1.0

        #borrowing constraint for C - guys. they just can't borrow in
        self.amin      = 0.0
        
        self.sim_time = 1000
        self.num_total_pop = 100000
        self.A        = 1.577707121233179 #this should give yc = 1 (approx.) z^2 case
        self.path_to_data_i_s = './input_data/data_i_s'

        



        self.agrid = np.load('./input_data/agrid.npy')
        self.epsgrid = np.load('./input_data/epsgrid.npy')    
        self.zgrid = np.load('./input_data/zgrid.npy')
        

        #conbined exogenous states
        #s = (e,z)'

        #pi(t,t+1)
        self.prob = np.load('./input_data/transition_matrix.npy')
    

        # ####do we need this one here?
        # #normalization to correct rounding error.
        # for i in range(prob.shape[0]):
        #     prob[i,:] = prob[i,:] / np.sum(prob[i,:])

        self.is_to_iz = np.load('./input_data/is_to_iz.npy')
        self.is_to_ieps = np.load('./input_data/is_to_ieps.npy')
        

       

        
    def __set_implied_parameters__(self):
        #length of grids
        self.num_a = len(self.agrid)
        self.num_eps = len(self.epsgrid)
        self.num_z = len(self.zgrid)
        self.num_s = self.prob.shape[0]

        
        #implied parameters
        self.nu = 1. - self.alpha - self.phi
        self.bh = self.beta*(1. + self.grate)**(self.eta*(1. - self.mu))  #must be less than one.
        self.varrho = (1. - self.alpha - self.nu)/(1. - self.alpha) * self.vthet / (self.vthet + self.veps)
    
        if self.bh >= 1.0:
            print('Error: bh must be in (0, 1) but bh = ', self.bh)
        
        
        
    def set_prices(self, p, rc):

        self.p = p
        self.rc = rc

        self.kcnc_ratio = ((self.theta * self.A)/(self.delk + self.rc))**(1./(1. - self.theta))
        self.w = (1. - self.theta)*self.A*self.kcnc_ratio**self.theta
        
        
        self.__is_price_set__ = True
        
        
        #implied prices
        self.rbar = (1. - self.taup) * self.rc
        self.rs = (1. - self.taup) * self.rc

        self.xi1 = ((self.ome*self.p)/(1. - self.ome))**(1./(self.rho-1.0))
        self.xi2 = (self.ome + (1. - self.ome) * self.xi1**self.rho)**(1./self.rho)
        self.xi3 = self.eta/(1. - self.eta) * self.ome * (1. - self.taun) / (1. + self.tauc) * self.w / self.xi2**self.rho
        
        self.xi8 = ((self.p*(self.nu**self.nu)*(self.alpha**(1.-self.alpha))/((self.w**self.nu)*((self.rs + self.delk)**(1.-self.nu)) )))**(1./(1.-self.alpha-self.nu))
        # self.xi8 = (self.alpha*self.p/(self.rs + self.delk))**(1./(1. - self.alpha))
        
        self.xi13 = self.nu/self.alpha*(self.rs + self.delk)/self.w


        self.denom = (1. + self.p*self.xi1)*(1. + self.tauc)
        self.xi4 = (1. + self.rbar) / self.denom
        self.xi5 = (1. + self.grate) / self.denom
        self.xi6 = (self.tran + self.yn - self.xnb) / self.denom
        self.xi7 = (1. - self.taun)*self.w/self.denom


        self.xi11 = (1. - self.taum) / self.denom
        self.xi10 = (self.p*self.xi8**self.alpha - (self.rs + self.delk)*self.xi8)*(1. - self.taum)/self.denom
        self.xi12 = self.vthet/self.veps*self.nu*self.p*self.xi8**self.alpha
        
        

    def print_parameters(self):
        
        print('')
        print('Parameters')
        print('alpha = ', self.alpha)
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
        print('tauc = ', self.tauc)
        print('taud = ', self.taud)
        print('taum = ', self.taum)
        print('taun = ', self.taun)
        print('taup = ', self.taup)
        print('theta = ', self.theta)
        print('tran (transfer) = ', self.tran)
        print('veps = ', self.veps)
        print('vthet = ', self.vthet)
        print('xnb = ', self.xnb)
        print('yn = ', self.yn)
        print('zeta = ', self.zeta)
        print('lbar = ', self.lbar)
        print('A = ', self.A)
        
        
        
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
            print('Implied Parameters')
            print('nu = ', self.nu)
            print('bh (beta_tilde) = ', self.bh)
            print('varrho = ', self.varrho)


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
            print('xi10 = ', self.xi10)
            print('xi11 = ', self.xi11)
            print('xi12 = ', self.xi12)
            
        else:
            print('')
            print('Prices not set')
            
        print('')
        print('Computational Parameters')

        print('amin = ', self.amin)
        print('sim_time = ', self.sim_time)
        print('num_total_pop = ', self.num_total_pop)

        
    def generate_util(self):
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
        taum = self.taum
        taun = self.taun
        taup = self.taup
        theta = self.theta
        tran = self.tran
        veps = self.veps
        vthet = self.vthet
        xnb = self.xnb
        yn = self.yn
        zeta= self.zeta

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

        xi1 = self.xi1
        xi2 = self.xi2
        xi3 = self.xi3
        xi4 = self.xi4
        xi5 = self.xi5
        xi6 = self.xi6
        xi7 = self.xi7
        xi8 = self.xi8
        xi13 = self.xi13
        xi10 = self.xi10
        xi11 = self.xi11
        xi12 = self.xi12
        ###end loading vars###
        
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
        taum = self.taum
        taun = self.taun
        taup = self.taup
        theta = self.theta
        tran = self.tran
        veps = self.veps
        vthet = self.vthet
        xnb = self.xnb
        yn = self.yn
        zeta= self.zeta

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

        xi1 = self.xi1
        xi2 = self.xi2
        xi3 = self.xi3
        xi4 = self.xi4
        xi5 = self.xi5
        xi6 = self.xi6
        xi7 = self.xi7
        xi8 = self.xi8
        xi13 = self.xi13
        xi10 = self.xi10
        xi11 = self.xi11
        xi12 = self.xi12
        ###end loading vars###
        
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

    import math    
        
        
    def generate_cstatic(self):
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
        taum = self.taum
        taun = self.taun
        taup = self.taup
        theta = self.theta
        tran = self.tran
        veps = self.veps
        vthet = self.vthet
        xnb = self.xnb
        yn = self.yn
        zeta= self.zeta

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
        num_s = self.prob.shape[0]

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
        xi13 = self.xi13
        xi10 = self.xi10
        xi11 = self.xi11
        xi12 = self.xi12
        #end loading 
        
        util = self.generate_util()
            
        @nb.jit(nopython = True)
        def get_cstatic(s):
            a = s[0]
            an = s[1]
            eps = s[2]

            u = -np.inf
            #u = -float('Inf')
            #u = -math.inf

            cc = -1.0
            cs = -1.0
            cagg = -1.0

            l = -1.0


            n = (xi3*eps - xi4*a + xi5*an - xi6)/(eps*(xi3 + xi7))

            if n < 0.0:
                n = 0.0


            if n >= 0. and n <= 1.:

                l = 1. - n

                #cc = xi3*eps*(1. - temp_n) #this is wrong at the corner.
                cc = xi4*a - xi5*an + xi6 + xi7*eps*n

                cs = xi1*cc
                cagg = xi2*cc
                u = util(cagg, 1. - n)


            return u, cc, cs, cagg, l ,n
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
        taum = self.taum
        taun = self.taun
        taup = self.taup
        theta = self.theta
        tran = self.tran
        veps = self.veps
        vthet = self.vthet
        xnb = self.xnb
        yn = self.yn
        zeta= self.zeta
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
        xi10 = self.xi10
        xi11 = self.xi11
        xi12 = self.xi12
        ###end loading vars###
        
        util = self.generate_util()


        

        @nb.jit(nopython = True)
        def get_sstatic(s):

            a = s[0]
            an = s[1]
            z = s[2]


            ks = z**(1./(1.-alpha-nu))*xi8
            ns = xi13*ks
            ys = z*(ks**alpha)*(ns**nu)

            cc = -1.0
            cs = -1.0
            cagg = -1.0


            #find cc
            #if feasible,... Here I really don't check an >= amin
            #cc = xi4*a - xi5*an + xi6 + xi11*(p*z*xi8**alpha - (rs+delk)*xi8)
                    
            cc = xi4*a - xi5*an + xi6 + xi11*(p*ys - (rs+delk)*ks - w*ns)
            cs = xi1*cc
            cagg = xi2 * cc

            u = -np.inf
            #adhoc feasbility check
            if (cagg > 0.0) and (an >= chi *ks):
                u = util(cagg, lbar)
                
            #else, return -np.inf


            # lbar is a given constant
            #mx, my, x are set to np.nan.
            return u, cc, cs, cagg, lbar, np.nan, np.nan, np.nan, ks, ys, ns
        
        return get_sstatic
    
    def get_policy(self):
        Econ = self

        #import data from Econ
        alpha = Econ.alpha
        beta = Econ.beta
        chi = Econ.chi
        delk = Econ.delk
        delkap = Econ.delkap
        eta = Econ.eta
        g = Econ.g
        grate = Econ.grate
        la = Econ.la
        mu = Econ.mu
        ome = Econ.ome
        phi = Econ.phi
        rho = Econ.rho
        tauc = Econ.tauc
        taud = Econ.taud
        taum = Econ.taum
        taun = Econ.taun
        taup = Econ.taup
        theta = Econ.theta
        tran = Econ.tran
        veps = Econ.veps
        vthet = Econ.vthet
        xnb = Econ.xnb
        yn = Econ.yn
        zeta= Econ.zeta

        agrid = Econ.agrid
        epsgrid = Econ.epsgrid
        zgrid = Econ.zgrid

        prob = Econ.prob

        is_to_iz = Econ.is_to_iz
        is_to_ieps = Econ.is_to_ieps

        amin = Econ.amin

        num_a = Econ.num_a
        num_eps = Econ.num_eps
        num_z = Econ.num_z
        num_s = Econ.num_s

        nu = Econ.nu
        bh = Econ.bh
        varrho = Econ.varrho

        w = Econ.w
        p = Econ.p
        rc = Econ.rc

        rbar = Econ.rbar
        rs = Econ.rs

        xi1 = Econ.xi1
        xi2 = Econ.xi2
        xi3 = Econ.xi3
        xi4 = Econ.xi4
        xi5 = Econ.xi5
        xi6 = Econ.xi6
        xi7 = Econ.xi7
        xi8 = Econ.xi8
        xi13 = Econ.xi13
        xi10 = Econ.xi10
        xi11 = Econ.xi11
        xi12 = Econ.xi12



        ###parameters for MPI###

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

        #to solve C-optimization problem, we need the max feasible set for an [amin, sup_an]
        c_supan = np.ones((num_a, num_eps)) * (-2.)
        for ia, a in enumerate(agrid):
                for ieps, eps in enumerate(epsgrid):

                    c_supan[ia, ieps] = ((1. + rbar)*a + (1. - taun)*w*eps + tran)/(1. + grate)


        get_sstatic = Econ.generate_sstatic()                    
        #to solve S-optimization problem, we need the max feasible set for an [amin, sup_an]                    
        s_supan = np.ones((num_a, num_z)) * (-2.)
        
       
        for iz, z in enumerate(zgrid):
            for ia, a in enumerate(agrid):

                ks = z**(1./(1.-alpha-nu))*xi8
                ns = xi13*ks
                
                ys = z*(ks**alpha)*(ns**nu)
                s_supan[ia, iz] = ((1. + rbar)*a + (1. - taum)*(p*ys - (rs + delk)*ks - w*ns) + tran + yn - xnb)/(1. + grate)

        del ks, ys, ns
                    
        # objective function of VFI optimization problem
        @nb.jit(nopython = True)    
        def _obj_loop_(*args):
            _an_ = args[0]
            _EV_ = args[1]
            _ia_ = args[2]
            _istate_ = args[3]
            _is_c_ = args[4]

            u = 0.0

            if _is_c_:
                u = get_cstatic(np.array([agrid[_ia_], _an_, epsgrid[is_to_ieps[_istate_]]]))[0]
            else:
                u = get_sstatic(np.array([agrid[_ia_], _an_, zgrid[is_to_iz[_istate_]]]))[0]

            return -(u + bh*fem_peval(_an_, agrid,  _EV_[0, :, _istate_])**(1. - mu))**(1./(1. - mu)) 
            # return -(u + fem_peval(_an_, agrid,  _EV_[0, :, _istate_] ))**(1./(1. - mu)) 
            
        #epsilon = np.finfo(float).eps
        @nb.jit(nopython = True)
        def _optimize_given_state_(_an_min_, _an_sup_, _EV_, _ia_ ,_istate_, _is_c_):        

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

            fx= _obj_loop_(x,  _EV_, _ia_ ,_istate_, _is_c_) 
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

                fu = _obj_loop_(u, _EV_, _ia_ ,_istate_, _is_c_)

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
        def _inner_loop_for_assigned_(assigned_indexes, _EV_, _vc_an_, _vcn_, _vc_util_, _is_c_):


            ibegin = assigned_indexes[0]
            iend = assigned_indexes[1]

            ind = 0
            for ipar_loop in range(ibegin, iend):

                #we should replace unravel_index_nb with something unflexible one.
                istate, ia = unravel_ip(ipar_loop)

                #this part may require modification
                if _is_c_:
                    an_sup = min(c_supan[ia, is_to_ieps[istate]] - 1.e-6, agrid[-1]) #no extrapolation for aprime
                    an_min = amin
                else:
                    an_sup = min(s_supan[ia, is_to_iz[istate]] - 1.e-6, agrid[-1]) #no extrapolation for aprime
                    an_min = chi*xi8

                ans =  _optimize_given_state_(an_min, an_sup, _EV_, ia, istate, _is_c_)
                

                _vc_an_[ind] = ans
                _vcn_[ind] = -_obj_loop_(ans, _EV_, ia, istate, _is_c_)

                if _is_c_:
                    _vc_util_[ind] = get_cstatic(np.array([agrid[ia], ans, epsgrid[is_to_ieps[istate]]]))[0]
                else:
                    _vc_util_[ind] = get_sstatic(np.array([agrid[ia], ans, zgrid[is_to_iz[istate]]]))[0]
                    
                ind = ind + 1

        @nb.jit(nopython = True)
        def _howard_iteration_(_vmax_, _vcn_, _vsn_, _vc_an_, _vc_util_, _vs_an_, _vs_util_, howard_iter):
            __EV__ = np.zeros((1, num_a,  num_s))
            if howard_iter > 0:

                for it_ho in range(howard_iter):

                    _vmax_[:] = np.fmax(_vcn_, _vsn_)

                    for ia in range(num_a):
                        __EV__[0, ia, :] = bh*((_vmax_[ia, :]**(1. - mu))@(prob.T)).reshape((1, 1, num_s)) 

                    for istate in range(num_s):
                        iz = is_to_iz[istate]
                        z = zgrid[iz]

                        for ia in range(num_a):

                            _vcn_[ia, istate] = (_vc_util_[ia, istate] + fem_peval(_vc_an_[ia, istate], agrid, __EV__[0, :, istate]) )**(1./(1. - mu))

                            _vsn_[ia, istate] = (_vs_util_[ia, istate] + fem_peval(_vs_an_[ia, istate], agrid, __EV__[0, :, istate]) )**(1./(1. - mu))
                    

        # ###pararell

        @nb.jit(nopython = True)
        def reshape_to_mat(v, val):
            for i in range(len(val)):
                # istate, ia, ikap = unravel_ip(i)
                # v[ia, ikap, istate] = val[i]
                
                istate, ia = unravel_ip(i)
                v[ia, istate] = val[i]
                


        #initialize variables for VFI            
        vc_an_tmp = np.ones((num_assigned))
        vcn_tmp = np.ones((num_assigned))
        vc_util_tmp = np.ones((num_assigned))

        vs_an_tmp = np.ones((num_assigned))
        vsn_tmp = np.ones((num_assigned))
        vs_util_tmp = np.ones((num_assigned))

        vc_an_full = None
        vcn_full = None
        vc_util_full = None

        if rank == 0:
            vc_an_full = np.ones((num_total_state))
            vcn_full = np.ones((num_total_state))*(-2.)
            vc_util_full = np.ones((num_total_state))

        vs_an_full = None
        vsn_full = None
        vs_util_full = None

        if rank == 0:
            vs_an_full = np.ones((num_total_state))
            vsn_full = np.ones((num_total_state))*(-2.)
            vs_util_full = np.ones((num_total_state))


        #all the dimensions are d
        vmax = np.ones((num_a, num_s))
        vmaxn = np.ones(vmax.shape)*100.0
        vmaxm1 = np.ones(vmax.shape)
        EV = np.ones((1,) + vmax.shape)


        vc_an = np.zeros(vmax.shape)
        vcn = np.ones(vmax.shape)*100.0
        vc_util = np.ones(vmax.shape)*100.0


        vs_an = np.zeros(vmax.shape)
        vsn = np.ones(vmax.shape)*100.0
        vs_util = np.ones(vmax.shape)*100.0

        max_iter = 50
        max_howard_iter = 100
        tol = 1.0e-8
        dist = 10000.0
        dist_sub = 10000.0
        it = 0

        ###record some time###
        t1, t2, t3, t4,tc1, tc2, ts1, ts2 = 0., 0., 0., 0., 0., 0., 0., 0.,
        if rank == 0:
            t1 = time.time()

        ###main VFI iteration###
        while it < max_iter and dist > tol:

            

            ### calculate EV and Bcast ###
            if rank == 0:
                it = it + 1
                # EV[:] = bh*((vmax**(1. - mu))@(prob.T)).reshape((1,) + vmax.shape)
                EV[:] = ((vmax)@(prob.T)).reshape((1,) + vmax.shape)                

            comm.Bcast([EV, MPI.DOUBLE])
            ### end calculate EV and Bcast ###            


            if rank == 0:
                tc1 = time.time()
            ###c-loop begins####
            _inner_loop_for_assigned_(assigned_state_range, EV, vc_an_tmp, vcn_tmp, vc_util_tmp,  True)
            ###c-loop ends####

            
            if rank == 0:
                tc2 = time.time()
                print('time for c = {:f}'.format(tc2 - tc1), end = ', ')


            if rank == 0:
                ts1 = time.time()
                
            ###s-loop begins####
            _inner_loop_for_assigned_(assigned_state_range, EV, vs_an_tmp, vsn_tmp, vs_util_tmp, False)
            ###s-loop ends####

            if rank == 0:
                ts2 = time.time()
                print('time for s = {:f}'.format(ts2 - ts1), end = ', ')


                

            comm.Gatherv(vcn_tmp,[vcn_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            comm.Gatherv(vsn_tmp,[vsn_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            comm.Gatherv(vc_an_tmp,[vc_an_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            comm.Gatherv(vs_an_tmp,[vs_an_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            comm.Gatherv(vc_util_tmp,[vc_util_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            comm.Gatherv(vs_util_tmp,[vs_util_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])

            if rank == 0:
                reshape_to_mat(vc_an, vc_an_full)
                reshape_to_mat(vs_an, vs_an_full)
                reshape_to_mat(vc_util, vc_util_full)
                reshape_to_mat(vs_util, vs_util_full)
                reshape_to_mat(vcn, vcn_full)
                reshape_to_mat(vsn, vsn_full)

                ####policy function iteration starts#####
                if max_howard_iter > 0:
                    #print('Starting Howard Iteration...')
                    t3 = time.time()
                    
                    _howard_iteration_(vmaxn, vcn, vsn, vc_an, vc_util, vs_an, vs_util, max_howard_iter)
                    
                    t4 = time.time()
                    print('time for HI = {:f}'.format(t4 - t3), end = ', ') 

                ####policy function iteration ends#####


            #### post_calc ####
            if rank == 0:


                pol_c = vcn > vsn
                vmaxn[:] = np.fmax(vcn, vsn)
                dist_sub = np.max(np.abs(vmaxn - vmax))
                dist = np.max(np.abs(vmaxn - vmax) / (1. + np.abs(vmaxn))) 
                print('{}th loop. dist = {:f}, dist_sub = {:f}.'.format(it, dist, dist_sub))
                vmaxm1[:] = vmax[:]
                vmax[:] = vmaxn[:]

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
        comm.Bcast([vc_an, MPI.DOUBLE])
        comm.Bcast([vs_an, MPI.DOUBLE])
        comm.Bcast([vcn, MPI.DOUBLE])
        comm.Bcast([vsn, MPI.DOUBLE])


        #return policy function
        self.vc_an = vc_an
        self.vs_an = vs_an
        self.vcn = vcn
        self.vsn = vsn

        
    #def get_obj(w, p, rc, vc_an, vs_an, vs_kapn, vcn, vsn):
    def simulate_model(self):
        Econ = self
        

        #import data from Econ
        alpha = Econ.alpha
        beta = Econ.beta
        chi = Econ.chi
        delk = Econ.delk
        delkap = Econ.delkap
        eta = Econ.eta
        g = Econ.g
        grate = Econ.grate
        la = Econ.la
        mu = Econ.mu
        ome = Econ.ome
        phi = Econ.phi
        rho = Econ.rho
        tauc = Econ.tauc
        taud = Econ.taud
        taum = Econ.taum
        taun = Econ.taun
        taup = Econ.taup
        theta = Econ.theta
        tran = Econ.tran
        veps = Econ.veps
        vthet = Econ.vthet
        xnb = Econ.xnb
        yn = Econ.yn
        zeta= Econ.zeta
        lbar = Econ.lbar

        agrid = Econ.agrid
        epsgrid = Econ.epsgrid
        zgrid = Econ.zgrid

        prob = Econ.prob

        is_to_iz = Econ.is_to_iz
        is_to_ieps = Econ.is_to_ieps

        amin = Econ.amin

        num_a = Econ.num_a
        num_eps = Econ.num_eps
        num_z = Econ.num_z
        num_s = Econ.num_s

        nu = Econ.nu
        bh = Econ.bh
        varrho = Econ.varrho

        w = Econ.w
        p = Econ.p
        rc = Econ.rc

        rbar = Econ.rbar
        rs = Econ.rs

        xi1 = Econ.xi1
        xi2 = Econ.xi2
        xi3 = Econ.xi3
        xi4 = Econ.xi4
        xi5 = Econ.xi5
        xi6 = Econ.xi6
        xi7 = Econ.xi7
        xi8 = Econ.xi8
        xi13 = Econ.xi13
        xi10 = Econ.xi10
        xi11 = Econ.xi11
        xi12 = Econ.xi12

        #simulation parameters
        sim_time = Econ.sim_time
        num_total_pop = Econ.num_total_pop

        #load the value functions

        vc_an = Econ.vc_an
        vs_an = Econ.vs_an
        vcn = Econ.vcn
        vsn = Econ.vsn    


        #obtain the value function and the discounted expected value function
        vn = np.fmax(vcn, vsn) #the value function
        # EV = bh*((vn**(1. - mu))@(prob.T)) # the discounted expected value function

        # ((vmax)@(prob.T)).reshape((1,) + vmax.shape)                
        EV = vn@(prob.T)


        @nb.jit(nopython = True)
        def unravel_ip(i_aggregated_state):

            istate, ia = unravel_index_nb(i_aggregated_state, num_s, num_a)
            #ia, ikap, istate = unravel_index_nb(i_aggregated_state, num_a, num_kap, num_s)
            return istate, ia, ikap

        get_cstatic = Econ.generate_cstatic()
        get_sstatic = Econ.generate_sstatic()


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


        

        #main data container
        data_a = None
        data_i_s = None
        data_is_c = None

        if rank == 0:
            data_a = np.ones((num_total_pop, sim_time)) * chi*xi8 #start from the feasible minimum value
            data_i_s = np.zeros((num_total_pop, sim_time), dtype = int)
            data_is_c = np.zeros((num_total_pop, sim_time), dtype = bool) 

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

        data_i_s_elem[:] = np.load(Econ.path_to_data_i_s + '_' + str(rank) + '.npy')

        
       
        # del data_i_s_import

        @nb.jit(nopython = True)
        def calc(data_a_, data_i_s_, data_is_c_):

            for t in range(1, sim_time):
                for i in range(num_pop_assigned):

                    a = data_a_[i, t-1]

                    istate = data_i_s_[i, t]
                    eps = epsgrid[is_to_ieps[istate]]
                    z = zgrid[is_to_iz[istate]]


                    #print('period = ', t)

                    an_c = femeval(a, agrid, vc_an[:,istate])
                    an_s = femeval(a, agrid, vs_an[:,istate])

                    # val_c = (get_cstatic([a, an_c, eps])[0]  + fem2d_peval(an_c, kapn_c, agrid, kapgrid, EV[:,:,istate])) **(1./(1.- mu))
                    # val_s = (get_sstatic([a, an_s, z])[0]    + fem2d_peval(an_s, kapn_s, agrid, kapgrid, EV[:,:,istate])) **(1./(1.- mu))


                    val_c = (get_cstatic([a, an_c, eps])[0]  + bh*fem_peval(an_c, agrid, EV[:,istate])**(1.-mu) ) **(1./(1.- mu))
                    val_s = (get_sstatic([a, an_s, z  ])[0]  + bh*fem_peval(an_s, agrid, EV[:,istate])**(1.-mu) ) **(1./(1.- mu))

                    if (val_c == val_s):
                        print('error: val_c == val_s')

                    i_c = val_c > val_s

                    an = i_c * an_c + (1. - i_c) * an_s

                    if (an < chi*xi8) and not i_c:
                        print('simulation error: an < k_s but S. t = ', t , ', i = ' , i)
                        
                    if (an < amin) and  i_c:
                        print('simulation error: an < amin but C. t = ', t , ', i = ' , i)
                        

                    data_a_[i, t] = an
                    data_is_c_[i, t] = i_c

        calc(data_a_elem, data_i_s_elem, data_is_c_elem)

        comm.Gatherv(data_a_elem, [data_a, all_num_pop_assigned, all_istart_pop_assigned,  MPI.DOUBLE.Create_contiguous(sim_time).Commit() ])

        comm.Gatherv(data_i_s_elem, [data_i_s, all_num_pop_assigned, all_istart_pop_assigned,  MPI.LONG.Create_contiguous(sim_time).Commit() ])   
        comm.Gatherv(data_is_c_elem, [data_is_c, all_num_pop_assigned, all_istart_pop_assigned,  MPI.BOOL.Create_contiguous(sim_time).Commit() ])


        #calculate other variables

        data_ss = None

        if rank == 0:

            data_ss = np.ones((num_total_pop, 17)) * (-2.0)

            t = -1
            for i in range(num_total_pop):

                #need to check the consistency within variables... there may be errors...
                if data_is_c[i, t]: 

                    a = data_a[i, t-1]
                    an = data_a[i, t]
                    eps = epsgrid[is_to_ieps[data_i_s[i, t]]]

                    data_ss[i,0] = 1.
                    data_ss[i,1] = a
                    data_ss[i,2] = np.nan
                    data_ss[i,3] = an
                    data_ss[i,4] = np.nan
                    data_ss[i,5] = eps
                    data_ss[i,6:11] = get_cstatic([a, an, eps])[1:]

                    #return u, cc, cs, cagg, l ,n

                else:

                    a = data_a[i, t-1]
                    an = data_a[i, t]
                    z = zgrid[is_to_iz[data_i_s[i, t]]]

                    data_ss[i,0] = 0.
                    data_ss[i,1] = a
                    data_ss[i,2] = np.nan
                    data_ss[i,3] = an
                    data_ss[i,4] = np.nan
                    data_ss[i,5] = z

                    
                    data_ss[i,6:17] = get_sstatic([a, an, z])[1:]

                    #return u, cc, cs, cagg, lbar, NAN, NAN, NAN, ks, ys, ns
        


        self.data_a = data_a
        self.data_i_s = data_i_s
        self.data_is_c = data_is_c
        self.data_ss = data_ss


        self.calc_moments()

        return

    def calc_age(self):

        #import data from Econ

        #simulation parameters
        sim_time = self.sim_time
        num_total_pop = self.num_total_pop

        #load main simlation result
        data_a = self.data_a
        data_i_s = self.data_i_s
        data_is_c = self.data_is_c
        data_ss = self.data_ss


        data_is_s = ~data_is_c
        
        s_age = np.ones(num_total_pop, dtype = int) * -1
        c_age = np.ones(num_total_pop, dtype = int) * -1

        for i in range(num_total_pop):
            if data_is_c[i,-1]:
                s_age[i] = -1
            else:
                t = 0
                while t < sim_time:
                    if data_is_s[i, -t - 1]:
                        s_age[i] = t
                    else:
                        break
                    t = t + 1


        for i in range(num_total_pop):
            if data_is_s[i,-1]:
                c_age[i] = -1
            else:
                t = 0
                while t < sim_time:
                    
                    if data_is_c[i, -t - 1]:
                        c_age[i] = t
        
                    else:
                        break
                    t = t + 1
        

        self.s_age = s_age
        self.c_age = c_age

        return
        
        
    def calc_moments(self):
        Econ = self

        #import data from Econ
        alpha = Econ.alpha
        beta = Econ.beta
        chi = Econ.chi
        delk = Econ.delk
        delkap = Econ.delkap
        eta = Econ.eta
        g = Econ.g
        grate = Econ.grate
        la = Econ.la
        mu = Econ.mu
        ome = Econ.ome
        phi = Econ.phi
        rho = Econ.rho
        tauc = Econ.tauc
        taud = Econ.taud
        taum = Econ.taum
        taun = Econ.taun
        taup = Econ.taup
        theta = Econ.theta
        tran = Econ.tran
        veps = Econ.veps
        vthet = Econ.vthet
        xnb = Econ.xnb
        yn = Econ.yn
        zeta= Econ.zeta
        lbar = Econ.lbar

        kcnc_ratio = Econ.kcnc_ratio


        A = Econ.A

        agrid = Econ.agrid
        epsgrid = Econ.epsgrid
        zgrid = Econ.zgrid

        prob = Econ.prob

        is_to_iz = Econ.is_to_iz
        is_to_ieps = Econ.is_to_ieps

        amin = Econ.amin

        num_a = Econ.num_a
        num_eps = Econ.num_eps
        num_z = Econ.num_z
        num_s = Econ.num_s

        nu = Econ.nu
        bh = Econ.bh
        varrho = Econ.varrho

        w = Econ.w
        p = Econ.p
        rc = Econ.rc

        rbar = Econ.rbar
        rs = Econ.rs

        xi1 = Econ.xi1
        xi2 = Econ.xi2
        xi3 = Econ.xi3
        xi4 = Econ.xi4
        xi5 = Econ.xi5
        xi6 = Econ.xi6
        xi7 = Econ.xi7
        xi8 = Econ.xi8
        xi13 = Econ.xi13
        xi10 = Econ.xi10
        xi11 = Econ.xi11
        xi12 = Econ.xi12

        #simulation parameters
        sim_time = Econ.sim_time
        num_total_pop = Econ.num_total_pop

        #load main simlation result
        data_a = self.data_a
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
            print('amax = {}'.format(np.max(data_a)))
            print('amin = {}'.format(np.min(data_a)))

            t = -1

            EIc = np.mean(data_ss[:,0])
            Ea  = np.mean(data_ss[:,1])
            Ecc = np.mean(data_ss[:,6])
            Ecs = np.mean(data_ss[:,7])
            El  = np.mean(data_ss[:,9])
            En  = np.mean( np.nan_to_num(data_ss[:,5]* data_ss[:,10] * data_ss[:,0] )) #num_to_num replace nan to zero.

            Eks = np.mean(data_ss[:,13] * (1. - data_ss[:,0]))
            Eys = np.mean(data_ss[:,14] * (1. - data_ss[:,0]))
            Ens = np.mean(data_ss[:,15] * (1. - data_ss[:,0]))                    


            Ecagg_c = np.mean((data_ss[:,6] + p*data_ss[:,7] )* (data_ss[:,0]))
            Ecagg_s = np.mean((data_ss[:,6] + p*data_ss[:,7] ) * (1. - data_ss[:,0]))

            ETn = np.mean(np.nan_to_num((taun*w*data_ss[:,5]*data_ss[:,10] - tran)*data_ss[:,0]) )
            # ETm = np.mean((taum*np.fmax(p*data_ss[:,14] - (rs + delk)*data_ss[:,13] - data_ss[:,12], 0.) - tran)*(1. - data_ss[:,0]) )
            ETm = np.mean(np.nan_to_num((taum*(p*data_ss[:,14] - (rs + delk)*data_ss[:,13] - w*data_ss[:,15]) - tran)*(1. - data_ss[:,0]) ))


            # yc = 1.0 #we can do this by choosing C-corp firm productivity A


            nc = En - Ens

            self.nc = nc
            self.En = En
            self.Ens = Ens

            kc = nc*kcnc_ratio            
                    
            # kc = ((w/(1. - theta)/A)**(1./theta))*nc
            yc = A * (kc**theta)*(nc**(1.-theta))
            yc_sub = Ecc + (grate + delk)*(kc + Eks) + g + xnb - yn

            Tc = tauc*(Ecc + p*Ecs)
            Tp = taup*(yc - w*nc - delk*kc)
            Td = taud*(yc - w*nc - (grate + delk)*kc - Tp)

            #b = (Tc + ETn + ETm + Tp + Td - g)/(rbar - grate) #old def
            b = Ea - (1. - taud)*kc - Eks
    #         netb = (grate + delk)*b ##typo
            netb = (rbar - grate)*b
            tax_rev = Tc + ETn + ETm + Td + Tp + tran

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

            mom0 = 1. - Ecs/Eys
            mom1 = 1. - (Ecc  + (grate + delk)*(kc + Eks) + g + xnb - yn)/yc
            mom2 = 1. - (tax_rev - E_transfer - netb)/g            
                    
            # # mom0 = 1. - (1. - theta)*yc/(w*nc)
            # mom0 = 1. - theta/(rc + delk) * yc/kc
            # mom1 = 1. - Ecs/Eys
            # mom2 = 1. - (tax_rev - tran - netb)/g
            # mom3 = 1. - (Ecc + (grate + delk)*(kc + Eks) + g + xnb - yn)/yc
            print('')


            # print('1-(1-thet)*yc/(E[w*eps*n]) = {}'.format(mom0))
            print('1-E(cs)/E(ys) = {}'.format(mom0))
            print('1-(Ecc+(grate+delk)*(kc + Eks)+ g + xnb - yn)/yc = {}'.format(mom1))            
            #print('1-((1-taud)kc+E(ks)+b)/Ea = {}'.format(1. - (b + (1.- taud)*kc + Eks)/Ea))
            print('1-(tax-tran-netb)/g = {}'.format(mom2))
                    
            # # print('1-(1-thet)*yc/(E[w*eps*n]) = {}'.format(mom0))
            # print('1-thet/(rc+delk)*yc/kc = {}'.format(mom0))
            # print('1-E(cs)/E(ys) = {}'.format(mom1))
            # #print('1-((1-taud)kc+E(ks)+b)/Ea = {}'.format(1. - (b + (1.- taud)*kc + Eks)/Ea))
            # print('1-(tax-tran-netb)/g = {}'.format(mom2))
            # print('1-(Ecc+(grate+delk)*(kc + Eks)+ g + xnb - yn)/yc = {}'.format(mom3))

            print('')
            print('Important Moments')
            print('  Financial assets(Ea) = {}'.format(Ea))
            print('  Sweat assets(Ekap) = {}'.format(np.nan))
            print('  Govt. debt(b) = {}'.format(b))
            print('  S-Corp Rental Capital (ks) =  {}'.format(Eks))
            print('  Ratio of C-corp workers(EIs) = {}'.format(EIc))
            print('  GDP(yc + yn + p*Eys) = {}'.format(GDP))

            print('')
            print('C-corporation production:')
            print('  Consumption(Ecc) = {}'.format(Ecc))
            print('  Investment((grate+delk)*kc) = {}'.format(kc*(grate + delk)))
            print('  Govt purchase(g) = {}'.format(g))
            print('  Output(yc) = {}'.format(yc))
            print('  Capital(kc) = {}'.format(kc))
            print('  Eff. Worked Hours in C (nc) = {}'.format(nc)) 


            print('')
            print('S-corporation production:')
            print('  Consumption(Ecs) = {}'.format(Ecs))
            print('  Output(Eys) = {}'.format(Eys))
            print('    Output in prices(p*Eys) = {}'.format(p*Eys))            
            print('  investment, sweat(Ex) = {}'.format(np.nan))
            print('  Hours, production(Emy) = {}'.format(np.nan))
            print('  Hours, sweat(Emx) = {}'.format(np.nan))
            print('  Sweat equity (kap) = {}'.format(np.nan))
            print('  Eff. Worked Hours(Ens) is S = {}'.format(Ens))             


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

            print('')
            print('National Product Shares (./GDP):')
            print('  Consumption(C) = {}'.format(C/GDP))
            print('  physical Investments(xc) = {}'.format((xc)/GDP))
            print('  physical Investments(Exs) = {}'.format((Exs)/GDP))
            print('  Govt Consumption(g) = {}'.format(g/GDP))
            print('  Nonbusiness investment(xnb) = {}'.format((xnb)/GDP))
            print('  sweat　 Investment(Ex) = {}'.format(np.nan))
            print('    Sum = {}'.format((C+xc+Exs+g+xnb)/GDP))
            #  print('    Sum = {}'.format((C+xc+Exs+g+xnb + Ex)/GDP))                    
                    
                    

            print('')
            print('Govt Budget:')
            print('  Public Consumption = {}'.format(g))
            print('  Net borrowing = {}'.format(netb))
            print('  Transfer = {}'.format(tran))
            print('  Tax revenue = {}'.format(tax_rev))

            print('')
            print('Gini Coefficients ***I HAVE NOT CHECK THIS PART***')
            print('  Financial Assets = {}'.format(gini(data_ss[:,1])))
            print('  Sweats Assets = {}'.format(np.nan))
            print('  C-wages = {}'.format(gini(np.nan_to_num(w*data_ss[:,5]* data_ss[:,10]*data_ss[:,0]))) )
            print('  S-income = {}'.format(gini((p*data_ss[:,14] ) * (1. - data_ss[:,0]))))
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
                print(' N is ', t, ', with = {:f}'.format((np.mean(np.all(data_is_c[:,-(t+2):-1] == False, axis = 1)) -\
                                                           np.mean(np.all(data_is_c[:,-(t+3):-1] == False, axis = 1)) )/ np.mean(1. - data_ss[:,0])))
                    


            print('')
            print('Labor Market')
            print('  Labor Supply(En)   = {}'.format(En))
            print('  Labor Demand of C(nc) = {}'.format(nc))
            print('  Labor Demand of S(Ens) = {}'.format(Ens))
            print('')

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

    #needs to be updated
    def calc_sweat_eq_value(self, discount = -1.0):
        Econ = self

        """
        This method solve the value function

        V(a, \kappa, s) = d(a, \kappa, s) + \hat{beta}E[u'_c/u_c * V(a', \kappa' ,s')]

        where d = dividend from sweat equity

        return d, val

        """

        #load variables
        alpha = Econ.alpha
        beta = Econ.beta
        chi = Econ.chi
        delk = Econ.delk
        delkap = Econ.delkap
        eta = Econ.eta
        g = Econ.g
        grate = Econ.grate
        la = Econ.la
        mu = Econ.mu
        ome = Econ.ome
        phi = Econ.phi
        rho = Econ.rho
        tauc = Econ.tauc
        taud = Econ.taud
        taum = Econ.taum
        taun = Econ.taun
        taup = Econ.taup
        theta = Econ.theta
        tran = Econ.tran
        veps = Econ.veps 
        vthet = Econ.vthet
        xnb = Econ.xnb
        yn = Econ.yn
        zeta= Econ.zeta
        lbar = Econ.lbar
        
        agrid = Econ.agrid
        epsgrid = Econ.epsgrid
        zgrid = Econ.zgrid

        prob = Econ.prob

        is_to_iz = Econ.is_to_iz
        is_to_ieps = Econ.is_to_ieps

        amin = Econ.amin

        num_a = Econ.num_a
        num_eps = Econ.num_eps
        num_z = Econ.num_z
        num_s = Econ.num_s

        nu = Econ.nu
        bh = Econ.bh
        varrho = Econ.varrho

        w = Econ.w
        p = Econ.p
        rc = Econ.rc

        rbar = Econ.rbar
        rs = Econ.rs

        xi1 = Econ.xi1
        xi2 = Econ.xi2
        xi3 = Econ.xi3
        xi4 = Econ.xi4
        xi5 = Econ.xi5
        xi6 = Econ.xi6
        xi7 = Econ.xi7
        xi8 = Econ.xi8
        xi13 = Econ.xi13
        xi10 = Econ.xi10
        xi11 = Econ.xi11
        xi12 = Econ.xi12

        #load the value functions
        vc_an = Econ.vc_an
        vs_an = Econ.vs_an
        vcn = Econ.vcn
        vsn = Econ.vsn  

        get_cstatic = Econ.generate_cstatic()
        get_sstatic = Econ.generate_sstatic()
        dc_util = Econ.generate_dc_util()

        #obtain the value function and the discounted expected value function
        vn = np.fmax(vcn, vsn) #the value function
        # EV = bh*((vn**(1. - mu))@(prob.T)) # the discounted expected value function
        EV = vn@(prob.T)


        ###obtain dividends and the stochastic discount factor###
        d = np.ones((num_a, num_kap, num_s)) * (-100.0)
    #     d_after_tax = np.ones((num_a, num_kap, num_s)) * (-2.0)

        u_c = np.zeros((num_a, num_kap, num_s))
        up_c = np.zeros((num_a, num_kap, num_s, num_s))


        an1 = np.zeros((num_a, num_kap, num_s))
        kapn1 = np.zeros((num_a, num_kap, num_s))

        an2 = np.zeros((num_a, num_kap, num_s, num_s))
        kapn2 = np.zeros((num_a, num_kap, num_s, num_s))

        to_be_s = np.zeros((num_a, num_kap, num_s), dtype = bool)

        #to be parallelized but nb.prange does not work here.
        @nb.jit(nopython = True)
        def _pre_calc_(d, u_c, up_c, an1, kapn1, an2, kapn2, to_be_s):
            for ia in range(num_a):
                a = agrid[ia]
    #         for ia, a in enumerate(agrid):
                for ikap, kap in enumerate(kapgrid): #need to be drop since we don't have kappa.
                    for istate in range(num_s):

                        an = None

                        an_s = vs_an[ia, istate]
                        z = zgrid[is_to_iz[istate]]

                        an_c = vc_an[ia, istate]
                        eps = epsgrid[is_to_ieps[istate]]

                        val_c = (get_cstatic([a, an_c, eps])[0]  + bh*fem2d_peval(an_c, agrid, EV[:,istate])**(1.-mu)) **(1./(1.- mu))
                        val_s = (get_sstatic([a, an_s, z  ])[0]  + bh*fem2d_peval(an_s, agrid, EV[:,istate])**(1.-mu)) **(1./(1.- mu))


                        if val_s >= val_c:
                            to_be_s[ia, istate] = True

                            an = an_s

                            u, cc, cs, cagg, l, mx, my, x, ks, ys, ns = get_sstatic([a, an, z])


                            #print(f'u = {u}')
                            u_c[ia, istate] = dc_util(cagg, l)

                            profit = p*ys - (rs + delk)*ks - w*ns #this can be nagative
                            tax = taum * max(profit, 0.) #do we need max?
                            div = phi * p * ys  #div related to sweat equity


                            d[ia, ikap, istate] = div
    #                         d_after_tax[ia, ikap, istate] = div - tax


                        else:
                            to_be_s[ia, istate] = False
                            d[ia, istate] = 0.0
    #                         d_after_tax[ia, ikap, istate] = 0.0

                            an = an_c
                            kapn = kapn_c

                            eps = epsgrid[is_to_ieps[istate]]

                            u, cc, cs, cagg, l ,n = get_cstatic([a, an, eps])
                            u_c[ia, ikap, istate] = dc_util(cagg, l)

                        an1[ia, ikap, istate] = an
                        kapn1[ia, ikap, istate] = kapn

                        for istate_n in range(num_s):

                            anp = None
                            kapnp = None

                            anp_s = fem2d_peval(an, kapn, agrid, kapgrid, vs_an[:, :, istate_n])
                            kapnp_s = fem2d_peval(an, kapn, agrid, kapgrid, vs_kapn[:, :, istate_n])
                            zp = zgrid[is_to_iz[istate_n]]

                            anp_c = fem2d_peval(an, kapn, agrid, kapgrid, vc_an[:, :, istate_n])
                            kapnp_c = la * kapn
                            epsp = epsgrid[is_to_ieps[istate_n]]

                            vcp = (get_cstatic([an, anp_c, epsp])[0]  + fem2d_peval(anp_c, kapnp_c, agrid, kapgrid, EV[:,:,istate_n])) **(1./(1.- mu))
                            vsp = (get_sstatic([an, anp_s, kapn, kapnp_s, zp])[0]    + fem2d_peval(anp_s, kapnp_s, agrid, kapgrid, EV[:,:,istate_n])) **(1./(1.- mu))


                            if vsp >= vcp:
                                anp = anp_s
                                kapnp = kapnp_s

                                u, cc, cs, cagg, l, mx, my, x, ks, ys, ns = get_sstatic([an, anp, kapn, kapnp, zp])
                                up_c[ia, ikap, istate, istate_n] = dc_util(cagg, l)

                                an2[ia, ikap, istate, istate_n] = anp
                                kapn2[ia, ikap, istate, istate_n] = kapnp



                            else:

                                anp = anp_c
                                kapnp = kapnp_c

                                u, cc, cs, cagg, l ,n = get_cstatic([an, anp, epsp])
                                up_c[ia, ikap, istate, istate_n] = dc_util(cagg, l)

                                an2[ia, ikap, istate, istate_n] = anp
                                kapn2[ia, ikap, istate, istate_n] = kapnp

        _pre_calc_(d, u_c, up_c, an1, kapn1, an2, kapn2, to_be_s)                        
        ###enc obtain dividends and the stochastic discount factor###


        @nb.jit(nopython = True, parallel = True)
        def _inner_get_sweat_equity_value_pp_(_d_, _val_, discount):
            bEval = np.zeros((num_a, num_kap, num_s))

            for ia in nb.prange(num_a):
                a = agrid[ia]
                for ikap in range(num_kap):
                    kap = kapgrid[ikap]
                    for istate in range(num_s):
                        dc_u = u_c[ia, ikap, istate]

                        for istate_n in range(num_s):

                            # an = an2[ia, ikap, istate, istate_n]
                            # kapn = kapn2[ia, ikap, istate, istate_n]

                            an = an1[ia, ikap, istate]
                            kapn = kapn1[ia, ikap, istate]

                            dc_un = up_c[ia, ikap, istate, istate_n]
                            val_p = fem2d_peval(an, kapn, agrid, kapgrid, _val_[:, :, istate_n])

                            # val_p[ia, ikap, istate, istate_n] = fem2d_peval(an, kapn, agrid, kapgrid, val[:, :, istate_n])

            
                            if discount < 0.0:
                                #use stochastic discount factor
                                bEval[ia, ikap, istate] += prob[istate, istate_n]  * val_p * bh* dc_un / dc_u
                            else:
                                #use the given discount factor
                                bEval[ia, ikap, istate] += prob[istate, istate_n]  * val_p * discount
                                
                            

                        #after taking expectation
                        bEval[ia, ikap, istate] =  bEval[ia, ikap, istate] 

            return _d_ + bEval

        val = d / (rbar - grate) #initial value
        val_tmp = np.zeros((num_a, num_kap, num_s)) #temporary storage

        dist = 1000.



        maxit = 1000
        tol = 1.0e-8
        it = 0

        t0 = time.time()    
        while it < maxit:
            it = it + 1
            val_tmp[:] = _inner_get_sweat_equity_value_pp_(d, val, discount)


            dist = np.max(np.abs(val_tmp - val))

            if dist < tol:
                break


            val[:] = val_tmp[:]

        t1 = time.time()

        if it >= maxit:
            print('Warning: iteration reached the max iteration.')
        print(f'elapsed time = {t1 - t0}')
        print(f'{it+1}th loop. dist = {dist}')

        
        if discount < 0.0:
            #save the result if discount factor is stochastic one.
            self.sweat_div = d
            self.sweat_val = val
            
        return d, val
        
        
    def simulate_other_vars(self):
        Econ = self

        """


        """

        #load variables
        alpha = Econ.alpha
        beta = Econ.beta
        chi = Econ.chi
        delk = Econ.delk
        delkap = Econ.delkap
        eta = Econ.eta
        g = Econ.g
        grate = Econ.grate
        la = Econ.la
        mu = Econ.mu
        ome = Econ.ome
        phi = Econ.phi
        rho = Econ.rho
        tauc = Econ.tauc
        taud = Econ.taud
        taum = Econ.taum
        taun = Econ.taun
        taup = Econ.taup
        theta = Econ.theta
        tran = Econ.tran
        veps = Econ.veps
        vthet = Econ.vthet
        xnb = Econ.xnb
        yn = Econ.yn
        zeta = Econ.zeta
        lbar = Econ.lbar

        agrid = Econ.agrid
        epsgrid = Econ.epsgrid
        zgrid = Econ.zgrid

        prob = Econ.prob

        is_to_iz = Econ.is_to_iz
        is_to_ieps = Econ.is_to_ieps

        amin = Econ.amin

        num_a = Econ.num_a
        num_eps = Econ.num_eps
        num_z = Econ.num_z
        num_s = Econ.num_s

        nu = Econ.nu
        bh = Econ.bh
        varrho = Econ.varrho

        w = Econ.w
        p = Econ.p
        rc = Econ.rc

        rbar = Econ.rbar
        rs = Econ.rs

        xi1 = Econ.xi1
        xi2 = Econ.xi2
        xi3 = Econ.xi3
        xi4 = Econ.xi4
        xi5 = Econ.xi5
        xi6 = Econ.xi6
        xi7 = Econ.xi7
        xi8 = Econ.xi8
        xi13 = Econ.xi13
        xi10 = Econ.xi10
        xi11 = Econ.xi11
        xi12 = Econ.xi12

        data_a = Econ.data_a
        data_i_s = Econ.data_i_s
        data_is_c = Econ.data_is_c

        #simulation parameters
        sim_time = Econ.sim_time
        num_total_pop = Econ.num_total_pop

        get_cstatic = Econ.generate_cstatic()
        get_sstatic = Econ.generate_sstatic()


        @nb.jit(nopython = True, parallel = True)
        def calc_all(data_a_, data_i_s_, data_is_c_,
                     data_u_, data_cc_, data_cs_, data_cagg_, data_l_, data_n_,  data_ks_, data_ys_, data_ns_):

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
                    mx = np.nan #redundant
                    my = np.nan #redundant
                    x = np.nan #redundant
                    ks = np.nan
                    ys = np.nan
                    ns = np.nan

                    a = data_a_[i, t-1]
                    an = data_a_[i, t]

                    is_c = data_is_c_[i, t]

                    if is_c:
                        u, cc, cs, cagg, l ,n = get_cstatic([a, an, eps])
                    else:
                        u, cc, cs, cagg, l, mx, my, x, ks, ys, ns = get_sstatic([a, an, z])

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
        data_ns = np.zeros(data_a.shape)                    

        #note that this does not store some impolied values,,,, say div or value of sweat equity
        calc_all(data_a, data_i_s, data_is_c, ##input
                 data_u, data_cc, data_cs, data_cagg, data_l, data_n, data_ks, data_ys, data_ns ##output
        )


        # for now I haven't programmed thsi part.
        # @nb.jit(nopython = True, parallel = True)
        # def calc_val_seq(data_a_, data_kap_, data_i_s_, data_is_c_, sweat_eq_val_ ,data_val_seq_):
        
        #     for t in nb.prange(1, sim_time):
        #         for i in range(num_total_pop):
            
        #             istate = data_i_s_[i, t]
        #             eps = epsgrid[is_to_ieps[istate]]
        #             z = zgrid[is_to_iz[istate]]
                
        #             a = data_a_[i, t-1]
        #             kap = data_kap_[i, t-1]
            
        #             an = data_a_[i, t]
        #             kapn = data_kap_[i, t]
                
        #         #             is_c = data_is_c_[i, t]
                
        #             data_val_seq_[i,t] = fem2d_peval(a, kap, agrid, kapgrid, sweat_eq_val_[:,:,istate])
                
        # sweat_div = self.sweat_div
        # sweat_val = self.sweat_val

        # sweat_val_bh = self.calc_sweat_eq_value(discount = self.bh)[1]
        # sweat_val_1gR = self.calc_sweat_eq_value(discount = (1. + self.grate)/(1. + self.rbar))[1]

        # self.sweat_val_bh = sweat_val_bh
        # self.sweat_val_1gR = sweat_val_1gR


        # data_div_sweat = np.zeros(data_a.shape)
        # data_val_sweat = np.zeros(data_a.shape)        
        # data_val_sweat_bh = np.zeros(data_a.shape)
        # data_val_sweat_1gR = np.zeros(data_a.shape)        

        

        # calc_val_seq(data_a, data_kap, data_i_s, data_is_c, sweat_div, data_div_sweat)
        # calc_val_seq(data_a, data_kap, data_i_s, data_is_c, sweat_val, data_val_sweat)
        # calc_val_seq(data_a, data_kap, data_i_s, data_is_c, sweat_val_bh, data_val_sweat_bh)
        # calc_val_seq(data_a, data_kap, data_i_s, data_is_c, sweat_val_1gR, data_val_sweat_1gR)                


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
            # np.save(dir_path_save + 'data_mx', self.data_mx[:, -100:])
            # np.save(dir_path_save + 'data_my', self.data_my[:, -100:])
            # np.save(dir_path_save + 'data_x', self.data_x[:, -100:])
            np.save(dir_path_save + 'data_ks', self.data_ks[:, -100:])
            np.save(dir_path_save + 'data_ys', self.data_ys[:, -100:])
            np.save(dir_path_save + 'data_ss', self.data_ss)

            np.save(dir_path_save + 'vc_an', self.vc_an)
            np.save(dir_path_save + 'vs_an', self.vs_an)
            # np.save(dir_path_save + 'vs_kapn', self.vs_kapn)
            np.save(dir_path_save + 'vcn', self.vcn)
            np.save(dir_path_save + 'vsn', self.vsn)


          

            # np.save(dir_path_save + 'sweat_div', self.sweat_div)
            # np.save(dir_path_save + 'sweat_val', self.sweat_val)
            # np.save(dir_path_save + 'sweat_val_bh', self.sweat_val_bh)            
            # np.save(dir_path_save + 'sweat_val_1gR', self.sweat_val_1gR)
            

            # np.save(dir_path_save + 'data_div_sweat', self.data_div_sweat[:, -100:])
            # np.save(dir_path_save + 'data_val_sweat', self.data_val_sweat[:, -100:])
            # np.save(dir_path_save + 'data_val_sweat_bh', self.data_val_sweat_bh[:, -100:])
            # np.save(dir_path_save + 'data_val_sweat_1gR', self.data_val_sweat_1gR[:, -100:])                        

            np.save(dir_path_save + 's_age', self.s_age)
            np.save(dir_path_save + 'c_age', self.c_age)

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
    econ.simulate_model()

    export_econ(econ)
