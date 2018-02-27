
#import Yuki's library in the directory ./library
import sys
sys.path.insert(0, './library/')


import numpy as np
import numba as nb
###usage
###@nb.jit(nopython = True)


#my library
#import
#from FEM import fem_peval #1D interpolation
from FEM_2D import fem2d_peval, fem2deval_mesh
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
                 agrid = None,
                 kapgrid = None,
                 epsgrid = None,
                 zgrid = None,
                 prob = None,
                 is_to_iz = None,
                 is_to_ieps = None,
                 amin = None,
                 num_suba_inner = None,
                 num_subkap_inner = None,
                 sim_time = None,
                 num_total_pop = None):
        

        
        
        self.__set_default_parameters__()
        
        #set the parameters if designated
        #I don't know how to automate these lines
        if alpha is not None: self.alpha = alpha
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
        if agrid is not None: self.agrid = agrid
        if kapgrid is not None: self.kapgrid = kapgrid
        if epsgrid is not None: self.epsgrid = epsgrid
        if zgrid is not None: self.zgrid = zgrid
        if prob is not None: self.prob = prob
        if is_to_iz is not None: self.is_to_iz = is_to_iz
        if is_to_ieps is not None: self.is_to_ieps = is_to_ieps
        if amin is not None: self.amin = amin
        if num_suba_inner is not None: self.num_suba_inner = num_suba_inner
        if num_subkap_inner is not None: self.num_subkap_inner = num_subkap_inner
        if sim_time is not None: self.sim_time = sim_time
        if num_total_pop is not None: self.num_total_pop = num_total_pop

        self.__set_implied_parameters__()
    
    def __set_default_parameters__(self):
        """
        Load the baseline value
        """
        
        self.__is_price_set__ = False
        self.alpha    = 0.4
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
        self.zeta     = 1.0
        self.sim_time = 1000
        self.num_total_pop = 100000




        self.agrid = np.load('./input_data/agrid.npy')
        self.kapgrid = np.load('./input_data/kapgrid.npy')
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
        
        #computational parameters
        self.amin      = 0.0
        self.num_suba_inner = 20
        self.num_subkap_inner = 30
        

        
    def __set_implied_parameters__(self):
        #length of grids
        self.num_a = len(self.agrid)
        self.num_kap = len(self.kapgrid)
        self.num_eps = len(self.epsgrid)
        self.num_z = len(self.zgrid)
        self.num_s = self.prob.shape[0]

        
        #implied parameters
        self.nu = 1. - self.alpha - self.phi
        self.bh = self.beta*(1. + self.grate)**(self.eta*(1. - self.mu))  #must be less than one.
        self.varrho = (1. - self.alpha - self.nu)/(1. - self.alpha) * self.vthet / (self.vthet + self.veps)
    
        if self.bh >= 1.0:
            print('Error: bh must be in (0, 1) but bh = ', self.bh)
        
        
        
    def set_prices(self, w, p, rc):
        self.w = w
        self.p = p
        self.rc = rc
        
        self.__is_price_set__ = True
        
        
        #implied prices
        self.rbar = (1. - self.taup) * self.rc
        self.rs = (1. - self.taup) * self.rc

        self.xi1 = ((self.ome*self.p)/(1. - self.ome))**(1./(self.rho-1.0))
        self.xi2 = (self.ome + (1. - self.ome) * self.xi1**self.rho)**(1./self.rho)
        self.xi3 = self.eta/(1. - self.eta) * self.ome * (1. - self.taun) / (1. + self.tauc) * self.w / self.xi2**self.rho
        self.xi8 = (self.alpha*self.p/(self.rs + self.delk))**(1./(1. - self.alpha))
        self.xi9 = self.eta / (1. - self.eta) * self.ome * self.p * self.nu                     * (1. - self.taum) / (1. + self.tauc) * self.xi8**self.alpha / self.xi2**self.rho

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
            print('xi9 = ', self.xi9)
            print('xi10 = ', self.xi10)
            print('xi11 = ', self.xi11)
            print('xi12 = ', self.xi12)
            
        else:
            print('')
            print('Prices not set')
            
        print('')
        print('Computational Parameters')

        print('amin = ', self.amin)
        print('num_suba_inner = ', self.num_suba_inner)
        print('num_subkap_inner = ', self.num_subkap_inner)
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
        kapgrid = self.kapgrid
        epsgrid = self.epsgrid
        zgrid = self.zgrid

        prob = self.prob

        is_to_iz = self.is_to_iz
        is_to_ieps = self.is_to_ieps

        amin = self.amin
        num_suba_inner = self.num_suba_inner
        num_subkap_inne = self.num_subkap_inner

        num_a = self.num_a
        num_kap = self.num_kap
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
        xi9 = self.xi9
        xi10 = self.xi10
        xi11 = self.xi11
        xi12 = self.xi12
        ###end loading vars###
        
        @nb.jit(nopython = True)
        def util(c, l):
            if c > 0.0 and l > 0.0 and l <= 1.0:
                return (1. - bh) * (((c**eta)*(l**(1. - eta)))**(1. - mu))
            else:
                return -np.inf

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
        kapgrid = self.kapgrid
        epsgrid = self.epsgrid
        zgrid = self.zgrid

        prob = self.prob

        is_to_iz = self.is_to_iz
        is_to_ieps = self.is_to_ieps

        amin = self.amin
        num_suba_inner = self.num_suba_inner
        num_subkap_inne = self.num_subkap_inner

        num_a = self.num_a
        num_kap = self.num_kap
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
        xi9 = self.xi9
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
        kapgrid = self.kapgrid
        epsgrid = self.epsgrid
        zgrid = self.zgrid

        prob = self.prob

        is_to_iz = self.is_to_iz
        is_to_ieps = self.is_to_ieps

        amin = self.amin
        num_suba_inner = self.num_suba_inner
        num_subkap_inne = self.num_subkap_inner

        num_a = self.num_a
        num_kap = self.num_kap
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
        xi9 = self.xi9
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

        agrid = self.agrid
        kapgrid = self.kapgrid
        epsgrid = self.epsgrid
        zgrid = self.zgrid

        prob = self.prob

        is_to_iz = self.is_to_iz
        is_to_ieps = self.is_to_ieps

        amin = self.amin
        num_suba_inner = self.num_suba_inner
        num_subkap_inne = self.num_subkap_inner

        num_a = self.num_a
        num_kap = self.num_kap
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
        xi9 = self.xi9
        xi10 = self.xi10
        xi11 = self.xi11
        xi12 = self.xi12
        ###end loading vars###
        
        util = self.generate_util()
        
        @nb.jit(nopython = True)
        def obj_find_mx(*args):
                mx = args[0]
                alp1 = args[1]
                alp2 = args[2]
                alp3 = args[3]

                return alp1*(1. - mx) - alp2*mx**((vthet + veps)/vthet) + alp3*mx

        @nb.jit(nopython = True)
        def d_obj_find_mx(*args):
                mx = args[0]
                alp1 = args[1]
                alp2 = args[2]
                alp3 = args[3]

                return -alp1 - alp2*((vthet + veps)/vthet)*mx**(veps/vthet) + alp3

        @nb.jit(nopython = True)
        def obj_find_mymax(*args):
                my = args[0]
                alp5 = args[1]
                return 1. - my - alp5*my**varrho

        @nb.jit(nopython = True)
        def d_obj_find_mymax(*args):
                my = args[0]
                alp5 = args[1]
                return  - 1.0 - varrho*alp5*my**(varrho - 1.0)


        @nb.jit(nopython = True)
        def obj_find_my(*args):
                my = args[0]
                alp1 = args[1]
                alp2 = args[2]
                alp3 = args[3]
                alp4 = args[4]
                alp5 = args[5]

                return alp1*(1. - my - alp5*my**varrho) - alp2 * my**(1. - nu/(1. - alpha)) - alp3*my + alp4*my**varrho       


        @nb.jit(nopython = True)
        def solve_mxmy(s):#return mx and my given (a, \kappa, a', \kappa'; z)
            a = s[0]
            an = s[1]
            kap = s[2]
            kapn = s[3]
            z = s[4]

            if kap == 0.0 and kapn > 0.0: 
                alp1 = eta/(1. - eta) * ome / xi2**rho / (1. + tauc)
                alp2 = vthet*(xi4*a - xi5*an + xi6)/veps/ ((1.+grate)*kapn/zeta)**(1./vthet)
                alp3 = vthet/(veps*denom)

                if alp2 == alp3:
                    return 1.0, 0.0 #in this case, utility must be -inf

                if (alp2< alp3) or (alp2 <=0):
                    return -1., -1. #the solution does not exist


                mx_lb = max( (alp3*vthet/(alp2*(vthet + veps)))**(vthet/veps), (alp3/alp2) )

        #         obj = lambda mx: alp1*(1. - mx) - alp2*mx**((vthet + veps)/vthet) + alp3*mx
        #         objprime = lambda mx: -alp1 - alp2*((vthet + veps)/vthet)*mx**(veps/vthet) + alp3
        #         ans = newton(obj_find_mx, mx_lb , fprime = d_obj_find_mx, args = (alp1, alp2, alp3), tol = 1.0e-15)


                ###start newton method
                mx = mx_lb
                it = 0
                maxit = 100 #scipy's newton use maxit = 50
                tol = 1.0e-15
                dist = 10000

                while it < maxit:
                    it = it + 1
                    res = alp1*(1. - mx) - alp2*mx**((vthet + veps)/vthet) + alp3*mx

                    dist = abs(res)

                    if dist < tol:
                        break

                    dres= -alp1 - alp2*((vthet + veps)/vthet)*mx**(veps/vthet) + alp3
                    diff = res/dres
                    mx = mx - res/dres


                #convergence check
                if it == maxit:
                    print('err: newton method for mx did not converge.')

                ans = mx    
                ###end newton method

                return ans, 0.

            elif kap == 0.0 and kapn == 0.0:

                #m=0.0, my = 0.0, x = 0.0
                #x = 0.0 should be designated outside of this function.
                return 0.0, 0.0


            elif kap > 0.0 and kapn > (1. - delkap)/(1. + grate) * kap: ##>=
                alp1 = xi9
                alp2 = (xi4 * a - xi5*an + xi6)/((z*kap**phi)**(1./(1.-alpha)))
                alp3 = xi10
                alp5 = (((((1. + grate)*kapn - (1. - delkap)*kap)/zeta)**(1./vthet))/(xi12 * (z*kap**phi)**(1./(1.-alpha))))**(vthet/(vthet + veps))
                alp4 = xi11 * xi12 * alp5

        #         obj_mymax = lambda my: 1. - my - alp5*my**varrho
        #         obj_mymax_prime = lambda my: - 1.0 - varrho*alp5*my**(varrho - 1.0)

        #         mymax = brentq(obj_find_mymax, 0., 1., args = (alp5,))


                ####bisection start
                mymax_lb = 0.
                mymax_ub = 1.

                #check bracketting
                val_lb = 1. - mymax_lb - alp5*mymax_lb**varrho
                val_ub = 1. - mymax_ub - alp5*mymax_ub**varrho

                if val_lb *val_ub > 0.0:
                    print('error: no bracket')
                sign = -1.0
                if val_ub > 0.:
                    sign = 1.0

                mymax = (mymax_lb + mymax_ub)/2.

                it = 0
                tol = 1.0e-12
                maxit = 200
                val_m = 10000
                while it < maxit:
                    it = it + 1
                    val_m = 1. - mymax - alp5*mymax**varrho


                    if sign * val_m > 0.:
                        mymax_ub = mymax
                    elif sign * val_m < 0.:
                        mymax_lb = mymax

                    diff = abs((mymax_lb + mymax_ub)/2 - mymax)
                    mymax = (mymax_lb + mymax_ub)/2.

                    if diff < tol:
                        break

                #convergence check
                if it == maxit:
                    print('err: bisection method for mymax did not converge.')
                    print('val_m = ', val_m)
                    print('mymax = ', mymax)
                ####bisection end

                # if mymax <= 0.0 or mymax >= 1.0:
                # do we need this part?
                if mymax < 0.0 or mymax > 1.0:
                    print('mymax is not well-solved or a corner solution')
                    return -1., -1.

        #         obj = lambda my: alp1*(1. - my - alp5*my**varrho) - alp2 * my**(1. - nu/(1. - alpha)) - alp3*my + alp4*my**varrho

                if obj_find_my(0.0, alp1, alp2, alp3, alp4, alp5)  == 0.0:
                    #print('my = 0.0')
                    return 0.0, 0.0

                if obj_find_my(mymax, alp1, alp2, alp3, alp4, alp5) == 0.0:
                    #print('my = mymax')
                    return alp5*mymax**varrho, mymax

                if obj_find_my(mymax, alp1, alp2, alp3, alp4, alp5) > 0:
                    #print('my does not exist')
                    return -1., -1.


        #          ans = brentq(obj_find_my, 0., mymax, args = (alp1, alp2, alp3, alp4, alp5), xtol=1e-20)


                ####bisection start
                my_lb = 0.
                my_ub = mymax

                #check bracketting
                val_lb = alp1*(1. - my_lb - alp5*my_lb**varrho) - alp2 * my_lb**(1. - nu/(1. - alpha)) - alp3*my_lb + alp4*my_lb**varrho 
                val_ub = alp1*(1. - my_ub - alp5*my_ub**varrho) - alp2 * my_ub**(1. - nu/(1. - alpha)) - alp3*my_ub + alp4*my_ub**varrho

                if val_lb *val_ub > 0.0:
                    print('error: no bracket')
                sign = -1.0
                if val_ub > 0.:
                    sign = 1.0

                my = (my_lb + my_ub)/2.

                it = 0
                tol = 1.0e-12
                rtol = 4.4408920985006262e-16
                maxit = 400
                val_m = 10000.
                while it < maxit:
                    it = it + 1
                    if my > 0. and my < 1.0e-6:
                        tol = 1.0e-20

                    val_m = alp1*(1. - my - alp5*my**varrho) - alp2 * my**(1. - nu/(1. - alpha)) - alp3*my + alp4*my**varrho  


                    if sign * val_m > 0.:
                        my_ub = my
                    elif sign * val_m < 0.:
                        my_lb = my

                    diff = abs((my_lb + my_ub)/2 - my)
                    my = (my_lb + my_ub)/2.

                    if diff < tol and abs(val_m) < rtol:
                        break

                #convergence check
                if it == maxit:
                    print('err: bisection method for my did not converge.')
                    #print('alp1 = ', alp1)
                    #print('alp2 = ', alp2)
                    #print('alp3 = ', alp3)
                    #print('alp4 = ', alp4)
                    #print('alp5 = ', alp5)
                    #print('val_m = ', val_m)
                    #print('my = ', my)
                    #print('mymax = ', mymax)

                ans = my
                ####bisection end


                if ans == 0.0:
                    print('A corner solution at 0.0 is obtianed: consider setting a smaller xtol.')
                    print('my = ', ans)


        #         if ans == mymax:
        #             #sometimes ans is extremely close to mymax
        #             #due to solver's accuracy.

                return alp5*ans**varrho, ans

            else:
                #print('error: kap < 0 is not allowed.')
                return -1., -1.


        @nb.jit(nopython = True)
        def get_sstatic(s):

            a = s[0]
            an = s[1]
            kap = s[2]
            kapn = s[3]
            z = s[4]


            u = -np.inf
            mx = -1.0
            my = -1.0
            l = -1.0
            x = -1.0
            cc = -1.0
            cs = -1.0
            cagg = -1.0
            ks = -1.0
            ys = -1.0


            if kapn >= (1. - delkap)/(1. + grate) * kap:
                mx, my = solve_mxmy(s)

                if mx >= 0. and my >= 0.:

                    x = -100.0

                    if mx == 0.0: #kap == 0.0 and kapn == 0.0 and my == 0.0:
                        x = 0.0
                    else:
                        x = ((((1. + grate)*kapn - (1. - delkap)*kap)/zeta)**(1./vthet))*mx**(-veps/vthet)


                    l = 1.0 - mx - my

                    if kap == 0.0:
                        #T^m if they do not operate
                        cc = xi4*a - xi5*an + xi6 - xi11*x / (1. - taum) + xi10*(z*kap**phi)**(1./(1.- alpha)) * my**(nu/(1. - alpha))
                    else:
                        cc = xi4*a - xi5*an + xi6 - xi11*x  + xi10*(z*kap**phi)**(1./(1.- alpha)) * my**(nu/(1. - alpha))


                    cs = xi1 * cc
                    cagg = xi2 * cc
                    ks = xi8 * (z*(kap**phi)* (my**nu)) ** (1./(1. - alpha))
                    ys = z*(ks**alpha)*(kap**phi)*(my**nu) #or (xi8**alpha)*(z*(kap**phi)*(my**nu))**(1./(1.- alpha))

                    #feasiblity check
                    #an >= p*chi*ys is working capital + financial constraint
                    if cagg > 0.0 and l > 0.0 and l <= 1.0 and an >= chi*p*ys:
                        u = util(cagg, l)

            return u, cc, cs, cagg, l, mx, my, x, ks, ys
        
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
        kapgrid = Econ.kapgrid
        epsgrid = Econ.epsgrid
        zgrid = Econ.zgrid

        prob = Econ.prob

        is_to_iz = Econ.is_to_iz
        is_to_ieps = Econ.is_to_ieps

        amin = Econ.amin
        num_suba_inner = Econ.num_suba_inner
        num_subkap_inner = Econ.num_subkap_inner

        num_a = Econ.num_a
        num_kap = Econ.num_kap
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
        xi9 = Econ.xi9
        xi10 = Econ.xi10
        xi11 = Econ.xi11
        xi12 = Econ.xi12



        ###parameters for MPI###

        num_total_state = num_a * num_kap * num_s
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

            istate, ia, ikap = unravel_index_nb(i_aggregated_state, num_s, num_a, num_kap)
            #ia, ikap, istate = unravel_index_nb(i_aggregated_state, num_a, num_kap, num_s)
            return istate, ia, ikap

        get_cstatic = Econ.generate_cstatic()

        #obtain the max...
        cvals_supan = np.ones((num_a, num_eps)) * (-2.)
        for ia, a in enumerate(agrid):
                for ieps, eps in enumerate(epsgrid):

                    cvals_supan[ia, ieps] = ((1. + rbar)*a + (1. - taun)*w*eps + tran)/(1. + grate)


        get_sstatic = Econ.generate_sstatic()



        ## construct subagrid and subkapgrid

        ## subagrid
        subagrid = np.ones((num_a-1)* (num_suba_inner-2) + num_a)
        num_suba = len(subagrid)

        ia = 0
        for ia in range(num_a - 1):
            subagrid[ia*(num_suba_inner) - ia : (ia+1)*(num_suba_inner) - ia] = np.linspace(agrid[ia], agrid[ia+1], num_suba_inner)



        ## subkapgrid
        subkapgrid = np.ones((num_kap - 1)* (num_subkap_inner - 2) + num_kap)

        num_subkap = len(subkapgrid)

        ikap = 0
        for ikap in range(num_kap-1):
            subkapgrid[ikap*(num_subkap_inner) - ikap : (ikap+1)*(num_subkap_inner) - ikap] = np.linspace(kapgrid[ikap], kapgrid[ikap+1], num_subkap_inner)



        ia_to_isuba = [1 for i in range(num_a)] #np.ones(num_a, dtype = int)
        for ia in range(num_a):
            ia_to_isuba[ia] = (num_suba_inner-1)*ia


        ikap_to_isubkap = [1 for i in range(num_kap)] #np.ones(num_kap, dtype = int)
        for ikap in range(num_kap):
            ikap_to_isubkap[ikap] = (num_subkap_inner-1)*ikap

        # ikap_to_isubkap_exp = [1 for i in range(num_kap+1)] #np.ones(num_kap, dtype = int)
        # for ikap in range(num_kap):
        #     ikap_to_isubkap_exp[ikap] = (num_subkap_inner-1)*ikap
        # ikap_to_isubkap_exp[-1] = len(subkapgrid) - 1



        # #check
        #subagrid[ia_to_isuba] - agrid
        #subkapgrid[ikap_to_isubkap] - kapgrid

        #export to numpy array,...sometimes we need this.
        ia_to_isuba = np.array(ia_to_isuba)
        ikap_to_isubkap = np.array(ikap_to_isubkap)
        #ikap_to_isubkap_exp = np.array(ikap_to_isubkap_exp)

        @nb.jit(nopython = True)
        def unravel_isub_mesh(i_subgrid_mesh):

            ind, ia_m, ikap_m = unravel_index_nb(i_aggregated_state, num_assigned, num_a-1, num_kap - 1)

            return ind, ia_m, ikap_m
        @nb.jit(nopython = True)
        def get_isub_mesh(ind, ia_m, ikap_m):

            return ind * (num_a-1)*(num_kap-1) + ia_m *(num_kap-1) + ikap_m



        #Store the S-corp utility values for the main grid

        s_util_origin = np.ones((num_assigned, num_a, num_kap))

        @nb.jit(nopython = True)
        def get_s_util_origin(_s_util_origin_ = s_util_origin):

            for ip in range(assigned_state_range[0], assigned_state_range[1]):

                ind = ip - assigned_state_range[0]
                istate, ia, ikap = unravel_ip(ip)


                for ian in range(num_a):
                    for ikapn in range(num_kap):

                        a = agrid[ia]
                        kap = kapgrid[ikap]
                        z = zgrid[is_to_iz[istate]]
                        an = agrid[ian]
                        kapn = kapgrid[ikapn]

                        state = [a, an, kap, kapn, z]

        #                 _s_util_origin_[ind, ian, ikapn] = get_sutil_cache(ip, ia_to_isuba[ian], ikap_to_isubkap[ikapn])
                        _s_util_origin_[ind, ian, ikapn] = get_sstatic(state)[0]
                        # get_sstatic(state)[0] #


        get_s_util_origin(s_util_origin)  


        #prepare for caching data
        num_prealloc = int(num_assigned * (num_a-1)* (num_kap - 1) * 0.05) #assign 5%
        num_cached = 0
        ind_s_util_finemesh_cached = np.ones((num_assigned * (num_a-1)* (num_kap - 1)), dtype = int)*(-1)
        s_util_finemesh_cached = np.zeros((num_prealloc, num_suba_inner, num_subkap_inner))


        #define inner loop functions

        @nb.jit(nopython = True)
        def _search_on_finer_grid_2_(ian_lo, ian_hi, ikapn_lo, ikapn_hi, _EV_, ip,
                                     _num_cached_,
                                     _ind_s_util_finemesh_cached_ = ind_s_util_finemesh_cached,
                                     _s_util_finemesh_cached_ = s_util_finemesh_cached):

            ian_c = ian_hi - 1
            ikapn_c = ikapn_hi - 1


            istate, ia, ikap = unravel_ip(ip)  
            ind = ip - assigned_state_range[0]

            a = agrid[ia]
            kap = kapgrid[ikap]
            z = zgrid[is_to_iz[istate]]

            subsubagrid = subagrid[ia_to_isuba[ian_lo] : ia_to_isuba[ian_hi]+1]
            subsubkapgrid = subkapgrid[ikap_to_isubkap[ikapn_lo] : ikap_to_isubkap[ikapn_hi]+1]

            if (len(subsubagrid) != 2*num_suba_inner - 1) or (len(subsubkapgrid) != 2*num_subkap_inner - 1):
                print('error: grid number of the finer grid')



            #define a finer grid

        #     s_util_fine_mesh = svals_util[ia, ikap, ia_to_isuba[ian_lo] : ia_to_isuba[ian_hi]+1, ikap_to_isubkap[ikapn_lo] : ikap_to_isubkap[ikapn_hi]+1 ,is_to_iz[istate]]

            s_util_fine_mesh = np.zeros((len(subsubagrid), len(subsubkapgrid)))


            for ian in [ian_lo, ian_hi-1]:
                for ikapn in [ikapn_lo, ikapn_hi-1]:
                    isub_mesh = get_isub_mesh(ind, ian, ikapn)

        #             print('isub_mesh = ', isub_mesh)

                    if _ind_s_util_finemesh_cached_[isub_mesh] == -1:


                        for ian_sub in range(ia_to_isuba[ian], ia_to_isuba[ian+1] + 1):
                            ian_ind = ian_sub - ia_to_isuba[ian_lo]

                            for ikapn_sub in range(ikap_to_isubkap[ikapn], ikap_to_isubkap[ikapn+1] + 1):
                                ikapn_ind = ikapn_sub - ikap_to_isubkap[ikapn_lo]

                                an = subagrid[ian_sub]
                                kapn = subkapgrid[ikapn_sub]

        #                         print('an = ', an)
        #                         print('kapn = ', kapn)

                                state = [a, an, kap, kapn, z]

                                s_util_fine_mesh[ian_ind, ikapn_ind] = get_sstatic(state)[0]

        #                         print('sutil = ', s_util_fine_mesh[ian_ind, ikapn_ind])

                        if _num_cached_ < num_prealloc:
                            ind_new_entry = _num_cached_  #this is inefficient. just keep track using another var.
                            #this should be less than something...
                            _s_util_finemesh_cached_[ind_new_entry, :, :] =                                         s_util_fine_mesh[(ia_to_isuba[ian] - ia_to_isuba[ian_lo]):(ia_to_isuba[ian+1]+1 - ia_to_isuba[ian_lo]),                                                          (ikap_to_isubkap[ikapn] - ikap_to_isubkap[ikapn_lo]):(ikap_to_isubkap[ikapn+1]+1 - ikap_to_isubkap[ikapn_lo])]

                            _ind_s_util_finemesh_cached_[isub_mesh] = ind_new_entry

                            _num_cached_ = _num_cached_ +1

        #                     print('cached')
        #                     print(_s_util_finemesh_cached_[ind_new_entry, :, :])
        #                     print('')
        #                     print('fine_mesh')
        #                     print(s_util_fine_mesh)





                    else: #if it is already cached

                         s_util_fine_mesh[(ia_to_isuba[ian] - ia_to_isuba[ian_lo]):(ia_to_isuba[ian+1]+1 - ia_to_isuba[ian_lo]),                          (ikap_to_isubkap[ikapn] - ikap_to_isubkap[ikapn_lo]):(ikap_to_isubkap[ikapn+1]+1 - ikap_to_isubkap[ikapn_lo])] =                             _s_util_finemesh_cached_[_ind_s_util_finemesh_cached_[isub_mesh], :, :]




            obj_fine_mesh = - (s_util_fine_mesh + fem2deval_mesh(subsubagrid, subsubkapgrid, agrid, kapgrid, _EV_[0, 0, :, :, istate])  )**(1./(1. - mu))




            ans_some = unravel_index_nb(np.argmin(obj_fine_mesh), len(subsubagrid), len(subsubkapgrid))



            _an_tmp_ = subsubagrid[ans_some[0]]
            _kapn_tmp_ = subsubkapgrid[ans_some[1]]
            _val_tmp_ = -obj_fine_mesh[ans_some[0], ans_some[1]] 
            _u_tmp_  = s_util_fine_mesh[ans_some[0], ans_some[1]]



            return ans_some[0], ans_some[1], _num_cached_ ,_an_tmp_, _kapn_tmp_, _val_tmp_, _u_tmp_


        @nb.jit(nopython = True)    
        def _inner_inner_loop_s_par_(ipar_loop, _EV_, _num_cached_): #, an_tmp, kapn_tmp, _val_tmp_, u_tmp):

            istate, ia, ikap = unravel_ip(ipar_loop)

        #     print('ia =, ', ia, ' ikap = ', ikap, ' istate = ', istate)

            a = agrid[ia]
            kap = kapgrid[ikap]

            iz = is_to_iz[istate]
            z = zgrid[iz]

            kapn_min = kap*(1. - delkap)/(1. + grate)

            #rough grid search 

            ind = ipar_loop - assigned_state_range[0]
            obj_mesh = - (s_util_origin[ind, :, :] + _EV_[0, 0, :, :, istate]) **(1./(1. - mu))


            ans_tmp = unravel_index_nb(np.argmin(obj_mesh), num_a, num_kap)


            an_tmp = agrid[ans_tmp[0]]
            kapn_tmp = kapgrid[ans_tmp[1]]
            val_tmp = -obj_mesh[ans_tmp[0], ans_tmp[1]] 
            u_tmp  = s_util_origin[ind, :, :][ans_tmp[0], ans_tmp[1]]


            #find surrounding grids
            ian_lo = max(ans_tmp[0] - 1, 0)
            ian_hi = min(ans_tmp[0] + 1, len(agrid) - 1)


            ikapn_lo = max(ans_tmp[1] - 1, 0)
            ikapn_hi = min(ans_tmp[1] + 1, len(kapgrid) - 1) #one is added



            if (ian_lo + 2 != ian_hi):
                if ian_lo == 0:
                    ian_hi = ian_hi + 1
                elif ian_hi == num_a -1:
                    ian_lo = ian_lo -1
                else:
                    print('error')

            if (ikapn_lo + 2 != ikapn_hi):
                if ikapn_lo == 0:
                    ikapn_hi = ikapn_hi + 1
                elif ikapn_hi == num_kap-1:
                    ikapn_lo = ikapn_lo - 1
                else:
                    print('error')



            #check if there exists mesh
            if (ian_lo + 2 != ian_hi) or (ikapn_lo + 2 != ikapn_hi):
                print('error: a finer grid was not generated.')


            if subkapgrid[ikap_to_isubkap[ikapn_hi]] <= kapn_min:
                print('subkapgrid[ikap_to_isubkap[ikapn_hi]] <= kapn_min')




            max_ian_sub = 2*num_suba_inner - 2
            max_ikapn_sub =  2*num_subkap_inner - 2


            max_iter = 100
            it_finer = 0
            while (it_finer < max_iter):
                it_finer = it_finer + 1


                ans = np.array([0, 0])
                ans[0], ans[1], _num_cached_, an_tmp, kapn_tmp, val_tmp, u_tmp =                      _search_on_finer_grid_2_(ian_lo, ian_hi, ikapn_lo, ikapn_hi, _EV_, ipar_loop, _num_cached_)    


                # move to an adjacent mesh or leave

                if ans[0] != 0 and ans[0] != max_ian_sub and ans[1] != 0 and ans[1] != max_ikapn_sub:
                    #solution
                    break
                elif ans[0] == 0 and (ans[1] != 0 or ans[1] != max_ikapn_sub):
                    if ian_lo == 0:
                        break
                    else:
                        ian_lo = ian_lo - 1
                        ian_hi = ian_hi - 1

                elif ans[0] == max_ian_sub and (ans[1] != 0 or ans[1] != max_ikapn_sub):
                    if ian_hi == num_a-1:
                        break
                    else:
                        ian_lo = ian_lo + 1
                        ian_hi = ian_hi + 1

                elif ans[1]  == 0 and (ans[0] != 0 or ans[0] != max_ian_sub):
                    if ikapn_lo == 0:
                        break
                    else:
                        ikapn_lo = ikapn_lo - 1
                        ikapn_hi = ikapn_hi - 1

                elif ans[1]  == max_ikapn_sub and (ans[0] != 0 or ans[0] != max_ian_sub):
                    if ikapn_hi == num_kap-1: 
                        break
                    else:
                        ikapn_lo = ikapn_lo + 1
                        ikapn_hi = ikapn_hi + 1

                elif ans[0] == 0 and ans[1] == 0:
                    if ian_lo == 0 and ikapn_lo == 0:
                        break
                    else:
                        if ian_lo != 0:
                            ian_lo = ian_lo - 1
                            ian_hi = ian_hi - 1
                        if ikapn_lo != 0:
                            ikapn_lo = ikapn_lo - 1
                            ikapn_hi = ikapn_hi - 1

                elif ans[0] == 0 and ans[1] == max_ikapn_sub:
                    if ian_lo == 0 and ikapn_hi == num_kap-1:
                        break
                    else:
                        if ian_lo != 0:
                            ian_lo = ian_lo - 1
                            ian_hi = ian_hi - 1
                        if ikapn_hi != num_kap - 1:
                            ikapn_lo = ikapn_lo + 1
                            ikapn_hi = ikapn_hi + 1

                elif ans[0] == max_ian_sub and ans[1] == 0:
                    if ian_hi == num_a-1 and ikapn_lo == 0:
                        break
                    else:
                        if ian_hi != num_a-1:
                            ian_lo = ian_lo + 1
                            ian_hi = ian_hi + 1
                        if ikapn_lo != 0:
                            ikapn_lo = ikapn_lo - 1
                            ikapn_hi = ikapn_hi - 1

                elif ans[0] == max_ian_sub and ans[1] == max_ikapn_sub:
                    if ian_hi == num_a-1 and ikapn_hi == num_kap-1:
                        break
                    else:
                        if ian_hi != num_a-1:
                            ian_lo = ian_lo + 1
                            ian_hi = ian_hi + 1
                        if ikapn_hi != num_kap-1:
                            ikapn_lo = ikapn_lo + 1
                            ikapn_hi = ikapn_hi + 1
                else:
                    print('error: ****')
        #                         print('ans[0] = {}, ans[1] = {}'.format(ans[0], ans[1]))

                if it_finer == max_iter:
                    print('error: reached the max fine grid search')
                    print('ia = ', ia, ', ikap = ', ikap, ', istate = ', istate)

                    break


            return an_tmp, kapn_tmp, val_tmp, u_tmp, _num_cached_


        @nb.jit(nopython = True) 
        def _inner_loop_s_with_range_(assigned_indexes, _EV_, _vs_an_, _vs_kapn_, _vsn_, _vs_util_, _num_cached_):


        #     for istate in range(num_s):
        #         for ia in range(num_a):
        #             for ikap in range(num_kap):

            ibegin = assigned_indexes[0]
            iend = assigned_indexes[1]



            ind = 0
            for ipar_loop in range(ibegin, iend):


                istate, ia, ikap = unravel_ip(ipar_loop)


                an_tmp = -3.0
                kapn_tmp = -3.0
                val_tmp = -3.0
                u_tmp = -3.0

                an_tmp, kapn_tmp, val_tmp, u_tmp, _num_cached_ = _inner_inner_loop_s_par_(ipar_loop, _EV_, _num_cached_)#, an_tmp, kapn_tmp, val_tmp, u_tmp)

                _vs_an_[ind] = an_tmp
                _vs_kapn_[ind] = kapn_tmp
                _vsn_[ind] = val_tmp
                _vs_util_[ind] = u_tmp



                ind = ind+1

            return _num_cached_

        @nb.jit(nopython = True)    
        def obj_loop_c(*args):
            _an_ = args[0]
            _EV_ = args[1]
            _ia_ = args[2]
            _ikap_ = args[3]
            _istate_ = args[4]

            u = get_cstatic(np.array([agrid[_ia_], _an_, epsgrid[is_to_ieps[_istate_]]]))[0]

            return -(u + fem2d_peval(_an_, la*kapgrid[_ikap_], agrid, kapgrid, _EV_[0, 0, :, :, _istate_]) )**(1./(1. - mu)) 

        #epsilon = np.finfo(float).eps
        @nb.jit(nopython = True)
        def _inner_inner_loop_c_(_an_sup_, _EV_, _ia_, _ikap_ ,_istate_):

             #arguments
            ax = 0.0
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
            fx= obj_loop_c(x,  _EV_, _ia_, _ikap_ ,_istate_) 
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
                    brent=fx


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

                fu = obj_loop_c(u, _EV_, _ia_, _ikap_ ,_istate_)

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
        def _inner_loop_c_with_range_(assigned_indexes, _EV_, _vc_an_, _vcn_, _vc_util_):

        #     for istate in range(num_s):
        #         for ia in range(num_a):
        #             for ikap in range(num_kap):

            ibegin = assigned_indexes[0]
            iend = assigned_indexes[1]

            ind = 0
            for ipar_loop in range(ibegin, iend):

                #we should replace unravel_index_nb with something unflexible one.
                istate, ia, ikap = unravel_ip(ipar_loop)

                an_sup = min(cvals_supan[ia, is_to_ieps[istate]] - 1.e-6, agrid[-1]) #no extrapolation for aprime

                ans = -10000.

                ans =  _inner_inner_loop_c_(an_sup, _EV_, ia, ikap ,istate)

                _vc_an_[ind] = ans
                _vcn_[ind] = -obj_loop_c(ans, _EV_, ia, ikap, istate)
        #         _vcn_[ind] = -obj_loop_c(ans, _EV_, ia, ikap, istate)
        # #         _vc_util_[ind] = get_cstatic([agrid[ia], ans, epsgrid[is_to_ieps[istate]]])[0]
                _vc_util_[ind] = get_cstatic(np.array([agrid[ia], ans, epsgrid[is_to_ieps[istate]]]))[0]

                ind = ind + 1


        @nb.jit(nopython = True)
        def _howard_iteration_(_vmax_, _vcn_, _vsn_, _vc_an_, _vc_util_,_vs_an_, _vs_kapn_, _vs_util_ ,howard_iter):
            __EV__ = np.zeros((1, 1, num_a, num_kap, num_s))
            if howard_iter > 0:

                for it_ho in range(howard_iter):

                    _vmax_[:] = np.fmax(_vcn_, _vsn_)
        #             _EV_[:] = bh*((_vmax_**(1. - mu))@(prob.T)).reshape((1, 1, num_a, num_kap, num_s)) #does not depend on ia or ikap
                    for ia in range(num_a):
                        __EV__[0, 0, ia, :, :] = bh*((_vmax_[ia,:,:]**(1. - mu))@(prob.T)).reshape((1, 1, 1, num_kap, num_s)) 


                    #EV[:] = bh*((vmaxn**(1. - mu))@(prob.T)).reshape((num_a, num_kap, num_s)) #does not depend on ia or ikap


                    for istate in range(num_s):
                        iz = is_to_iz[istate]
                        z = zgrid[iz]
                        #EV_c_interp_f = RectBivariateSpline(agrid, kapgrid, EV[0, 0, :, :, istate],kx = 1, ky = 1)

                        for ia in range(num_a):
                            for ikap in range(num_kap):
                                #kap = kapgrid[ikap]
                                #update C

        #                         EV_c = EV_c_interp_f(agrid, la*kapgrid[ikap])
        #                         obj2 = interp1d(agrid,  EV_c.reshape(num_a), fill_value = 'extrapolate')

                                #obj = lambda x: (get_util_c(np.array([agrid[ia], x, epsgrid[is_to_ieps[istate]]])) + obj2(x))**(1./(1. - mu))
                                #vcn[ia, ikap, istate] = (vc_util[ia, ikap, istate] + obj2(vc_an[ia, ikap, istate]))**(1./(1. - mu))
                                _vcn_[ia, ikap, istate] = (_vc_util_[ia, ikap, istate] +                                                          fem2d_peval(_vc_an_[ia, ikap, istate], la*kapgrid[ikap], agrid, kapgrid, __EV__[0,0,:,:,istate])                                                          )**(1./(1. - mu))




                                #update S

                                #vsn[ia, ikap, istate] = (vs_util[ia, ikap, istate] + EV_interp_f(vs_an[ia, ikap, istate] , vs_kapn[ia, ikap, istate]))**(1./(1. - mu))
                                _vsn_[ia, ikap, istate] = (_vs_util_[ia, ikap, istate] +                                                          fem2d_peval(_vs_an_[ia, ikap, istate] , _vs_kapn_[ia, ikap, istate], agrid, kapgrid, __EV__[0,0,:,:,istate])                                                          )**(1./(1. - mu))

                #after 


        #             vmaxn[:] = np.fmax(vcn, vsn)


        ###pararell

        @nb.jit(nopython = True)
        def reshape_to_mat(v, val):
            for i in range(len(val)):
                istate, ia, ikap = unravel_ip(i)
                v[ia, ikap, istate] = val[i]


        #initialize variables for VFI            
        vc_an_tmp = np.ones((num_assigned))
        vcn_tmp = np.ones((num_assigned))
        vc_util_tmp = np.ones((num_assigned))

        vs_an_tmp = np.ones((num_assigned))
        vs_kapn_tmp = np.ones((num_assigned))
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
        vs_kapn_full = None
        vsn_full = None
        vs_util_full = None

        if rank == 0:
            vs_an_full = np.ones((num_total_state))
            vs_kapn_full = np.ones((num_total_state))
            vsn_full = np.ones((num_total_state))*(-2.)
            vs_util_full = np.ones((num_total_state))



        vmax = np.ones((num_a, num_kap, num_s))
        vmaxn = np.ones((num_a, num_kap, num_s))*100.0
        vmaxm1 = np.ones(vmax.shape)
        EV = np.ones((1, 1, num_a, num_kap, num_s))


        vc_an = np.zeros((num_a, num_kap, num_s))
        vcn = np.ones((num_a, num_kap, num_s))*100.0
        vc_util = np.ones((num_a, num_kap, num_s))*100.0


        vs_an = np.zeros((num_a, num_kap, num_s))
        vs_kapn = np.zeros((num_a, num_kap, num_s))
        vsn = np.ones((num_a, num_kap, num_s))*100.0
        vs_util = np.ones((num_a, num_kap, num_s))*100.0

        max_iter = 50
        max_howard_iter = 200
        tol = 1.0e-5
        dist = 10000.0
        dist_sub = 10000.0
        it = 0

        ###record some time###
        t1, t2, t3, t4,tc1, tc2, ts1, ts2 = 0., 0., 0., 0., 0., 0., 0., 0.,
        if rank == 0:
            t1 = time.time()

        ###main VFI iteration###
        while it < max_iter and dist > tol:



            if rank == 0:
                it = it + 1
                EV[:] = bh*((vmax**(1. - mu))@(prob.T)).reshape((1, 1, num_a, num_kap, num_s))

            comm.Bcast([EV, MPI.DOUBLE])


            if rank == 0:
                tc1 = time.time()
            ###c-loop begins####
            _inner_loop_c_with_range_(assigned_state_range, EV, vc_an_tmp, vcn_tmp, vc_util_tmp)


            ###c-loop ends####
            if rank == 0:
                tc2 = time.time()
                print('time for c = {:f}'.format(tc2 - tc1), end = ', ')


            if rank == 0:
                ts1 = time.time()
            ###s-loop begins####

            num_cached = _inner_loop_s_with_range_(assigned_state_range, EV, vs_an_tmp ,vs_kapn_tmp, vsn_tmp, vs_util_tmp, num_cached)


            ###s-loop ends####

            if rank == 0:
                ts2 = time.time()
                print('time for s = {:f}'.format(ts2 - ts1), end = ', ')


            ####policy function iteration starts#####

            comm.Gatherv(vcn_tmp,[vcn_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            comm.Gatherv(vsn_tmp,[vsn_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            comm.Gatherv(vc_an_tmp,[vc_an_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            comm.Gatherv(vs_an_tmp,[vs_an_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            comm.Gatherv(vs_kapn_tmp,[vs_kapn_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            comm.Gatherv(vc_util_tmp,[vc_util_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])
            comm.Gatherv(vs_util_tmp,[vs_util_full, all_num_assigned, all_istart_assigned ,MPI.DOUBLE])

            if rank == 0:
                reshape_to_mat(vc_an, vc_an_full)
                reshape_to_mat(vs_an, vs_an_full)
                reshape_to_mat(vs_kapn, vs_kapn_full)
                reshape_to_mat(vc_util, vc_util_full)
                reshape_to_mat(vs_util, vs_util_full)
                reshape_to_mat(vcn, vcn_full)
                reshape_to_mat(vsn, vsn_full)

                if max_howard_iter > 0:
                    #print('Starting Howard Iteration...')
                    t3 = time.time()

                    _howard_iteration_(vmaxn, vcn, vsn, vc_an, vc_util, vs_an, vs_kapn, vs_util ,max_howard_iter)

                if max_howard_iter > 0:
                    t4 = time.time()
                    print('time for HI = {:f}'.format(t4 - t3), end = ', ') 

            ####policy function iteration ends#####


            ####post_calc
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
        comm.Bcast([vs_kapn, MPI.DOUBLE])
        comm.Bcast([vcn, MPI.DOUBLE])
        comm.Bcast([vsn, MPI.DOUBLE])


        #return policy function
        self.vc_an = vc_an
        self.vs_an = vs_an
        self.vs_kapn = vs_kapn
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

        agrid = Econ.agrid
        kapgrid = Econ.kapgrid
        epsgrid = Econ.epsgrid
        zgrid = Econ.zgrid

        prob = Econ.prob

        is_to_iz = Econ.is_to_iz
        is_to_ieps = Econ.is_to_ieps

        amin = Econ.amin
        num_suba_inner = Econ.num_suba_inner
        num_subkap_inner = Econ.num_subkap_inner

        num_a = Econ.num_a
        num_kap = Econ.num_kap
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
        xi9 = Econ.xi9
        xi10 = Econ.xi10
        xi11 = Econ.xi11
        xi12 = Econ.xi12

        #simulation parameters
        sim_time = Econ.sim_time
        num_total_pop = Econ.num_total_pop

        #load the value functions

        vc_an = Econ.vc_an
        vs_an = Econ.vs_an
        vs_kapn = Econ.vs_kapn
        vcn = Econ.vcn
        vsn = Econ.vsn    


        #obtain the value function and the discounted expected value function
        vn = np.fmax(vcn, vsn) #the value function
        EV = bh*((vn**(1. - mu))@(prob.T)) # the discounted expected value function


        @nb.jit(nopython = True)
        def unravel_ip(i_aggregated_state):

            istate, ia, ikap = unravel_index_nb(i_aggregated_state, num_s, num_a, num_kap)
            #ia, ikap, istate = unravel_index_nb(i_aggregated_state, num_a, num_kap, num_s)
            return istate, ia, ikap

        get_cstatic = Econ.generate_cstatic()
        get_sstatic = Econ.generate_sstatic()

        #do we need this one here...?
        cvals_supan = np.ones((num_a, num_eps)) * (-2.)
        for ia, a in enumerate(agrid):
                for ieps, eps in enumerate(epsgrid):

                    cvals_supan[ia, ieps] = ((1. + rbar)*a + (1. - taun)*w*eps + tran)/(1. + grate)


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
        data_kap_elem = np.ones((num_pop_assigned, sim_time))*0.0
        data_i_s_elem = np.ones((num_pop_assigned, sim_time), dtype = int)*7
        data_is_c_elem = np.zeros((num_pop_assigned, sim_time), dtype = bool) 
        data_is_c_elem[0:int(num_pop_assigned*0.7), 0] = True

        #main data container
        data_a = None
        data_kap = None
        data_i_s = None
        data_is_c = None

        if rank == 0:
            data_a = np.zeros((num_total_pop, sim_time))
            data_kap = np.zeros((num_total_pop, sim_time))
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


        ###load shock data###
        data_i_s_import = np.load('./input_data/data_i_s.npy')
        data_i_s_elem[:] = data_i_s_import[assigned_pop_range[0]:assigned_pop_range[1],0:sim_time]
       
        del data_i_s_import

        @nb.jit(nopython = True)
        def calc(data_a_, data_kap_, data_i_s_, data_is_c_):

            for t in range(1, sim_time):
                for i in range(num_pop_assigned):

                    a = data_a_[i, t-1]
                    kap = data_kap_[i, t-1]

                    istate = data_i_s_[i, t]
                    eps = epsgrid[is_to_ieps[istate]]
                    z = zgrid[is_to_iz[istate]]


                    #print('period = ', t)

                    an_c = fem2d_peval(a, kap, agrid, kapgrid, vc_an[:,:,istate])
                    kapn_c = la*kap #fem2d_peval(a, kap, agrid, kapgrid, vc_kapn[:,:,istate]) #or lambda * kap
                    #kapn_c = fem2d_peval(a, kap, agrid, kapgrid, vc_kapn[:,:,istate])

                    an_s = fem2d_peval(a, kap, agrid, kapgrid, vs_an[:,:,istate])
                    kapn_s = fem2d_peval(a, kap, agrid, kapgrid, vs_kapn[:,:,istate])
                    #if we dont 'want to allow for extraplation
                    #kapn_s = max((1. - delkap)/(1.+grate)*kap, fem2d_peval(a, kap, agrid, kapgrid, vs_kapn[:,:,istate]))


                    val_c = (get_cstatic([a, an_c, eps])[0]  + fem2d_peval(an_c, kapn_c, agrid, kapgrid, EV[:,:,istate])) **(1./(1.- mu))
                    val_s = (get_sstatic([a, an_s, kap, kapn_s, z])[0]    + fem2d_peval(an_s, kapn_s, agrid, kapgrid, EV[:,:,istate])) **(1./(1.- mu))

                    if (val_c == val_s):
                        print('error: val_c == val_s')

                    i_c = val_c > val_s

                    an = i_c * an_c + (1. - i_c) * an_s
                    kapn = i_c * kapn_c + (1. - i_c) * kapn_s

                    data_a_[i, t] = an
                    data_kap_[i, t] = kapn
                    data_is_c_[i, t] = i_c

        calc(data_a_elem, data_kap_elem, data_i_s_elem, data_is_c_elem)

        comm.Gatherv(data_a_elem, [data_a, all_num_pop_assigned, all_istart_pop_assigned,  MPI.DOUBLE.Create_contiguous(sim_time).Commit() ])
        comm.Gatherv(data_kap_elem, [data_kap, all_num_pop_assigned, all_istart_pop_assigned,  MPI.DOUBLE.Create_contiguous(sim_time).Commit() ])
        comm.Gatherv(data_i_s_elem, [data_i_s, all_num_pop_assigned, all_istart_pop_assigned,  MPI.LONG.Create_contiguous(sim_time).Commit() ])   
        comm.Gatherv(data_is_c_elem, [data_is_c, all_num_pop_assigned, all_istart_pop_assigned,  MPI.BOOL.Create_contiguous(sim_time).Commit() ])


        #calculate other variables

        data_ss = None

        if rank == 0:

            data_ss = np.ones((num_total_pop, 15)) * (-2.0)

            t = -1
            for i in range(num_total_pop):

                #need to check the consistency within variables... there may be errors...
                if data_is_c[i, t]: 

                    a = data_a[i, t-1]
                    kap = data_kap[i, t-1]
                    an = data_a[i, t]
                    kapn = data_kap[i, t]
                    eps = epsgrid[is_to_ieps[data_i_s[i, t]]]

                    data_ss[i,0] = 1.
                    data_ss[i,1] = a
                    data_ss[i,2] = kap
                    data_ss[i,3] = an
                    data_ss[i,4] = kapn
                    data_ss[i,5] = eps
                    data_ss[i,6:11] = get_cstatic([a, an, eps])[1:]

                else:

                    a = data_a[i, t-1]
                    kap = data_kap[i, t-1]
                    an = data_a[i, t]
                    kapn = data_kap[i, t]
                    z = zgrid[is_to_iz[data_i_s[i, t]]]

                    data_ss[i,0] = 0.
                    data_ss[i,1] = a
                    data_ss[i,2] = kap
                    data_ss[i,3] = an
                    data_ss[i,4] = kapn
                    data_ss[i,5] = z
                    data_ss[i,6:15] = get_sstatic([a, an, kap, kapn, z])[1:]
        


        self.data_a = data_a
        self.data_kap = data_kap
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
        data_kap = self.data_kap
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

        agrid = Econ.agrid
        kapgrid = Econ.kapgrid
        epsgrid = Econ.epsgrid
        zgrid = Econ.zgrid

        prob = Econ.prob

        is_to_iz = Econ.is_to_iz
        is_to_ieps = Econ.is_to_ieps

        amin = Econ.amin
        num_suba_inner = Econ.num_suba_inner
        num_subkap_inner = Econ.num_subkap_inner

        num_a = Econ.num_a
        num_kap = Econ.num_kap
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
        xi9 = Econ.xi9
        xi10 = Econ.xi10
        xi11 = Econ.xi11
        xi12 = Econ.xi12

        #simulation parameters
        sim_time = Econ.sim_time
        num_total_pop = Econ.num_total_pop

        #load main simlation result
        data_a = self.data_a
        data_kap = self.data_kap
        data_i_s = self.data_i_s
        data_is_c = self.data_is_c
        data_ss = self.data_ss


        #print moments

        mom0 = None
        mom1 = None
        mom2 = None
        mom3 = None

        if rank == 0:
            print('amax = {}'.format(np.max(data_a)))
            print('amin = {}'.format(np.min(data_a)))
            print('kapmax = {}'.format(np.max(data_kap)))
            print('kapmin = {}'.format(np.min(data_kap)))

            t = -1

            EIc = np.mean(data_ss[:,0])
    #         Ea = np.mean(data_ss[:,3])
            Ea = np.mean(data_ss[:,1])
    #         Ekap = np.mean(data_ss[:,4])
            Ekap = np.mean(data_ss[:,2])
            Ecc = np.mean(data_ss[:,6])
            Ecs = np.mean(data_ss[:,7])
            El = np.mean(data_ss[:,9])
            En = np.mean(data_ss[:,5]* data_ss[:,10] * (data_ss[:,0]))
        #     En = np.mean(data_ss[:,10] * (data_ss[:,0]))
            Ex = np.mean(data_ss[:,12] * (1. - data_ss[:,0]))
            Eks = np.mean(data_ss[:,13] * (1. - data_ss[:,0]))
            Eys = np.mean(data_ss[:,14] * (1. - data_ss[:,0]))
            Emx = np.mean(data_ss[:,10] * (1. - data_ss[:,0]))
            Emy = np.mean(data_ss[:,11] * (1. - data_ss[:,0]))


            Ecagg_c = np.mean((data_ss[:,6] + p*data_ss[:,7] )* (data_ss[:,0]))
            Ecagg_s = np.mean((data_ss[:,6] + p*data_ss[:,7] ) * (1. - data_ss[:,0]))

            ETn = np.mean((taun*w*data_ss[:,5]*data_ss[:,10] - tran)*data_ss[:,0])
            ETm = np.mean((taum*np.fmax(p*data_ss[:,14] - (rs + delk)*data_ss[:,13] - data_ss[:,12], 0.) - tran)*(1. - data_ss[:,0]) )


            yc = 1.0 #we can do this by choosing C-corp firm productivity A


            nc = En
            kc = theta*yc/(rc + delk)

            yc_sub = Ecc  + Ex+ (grate + delk)*(kc + Eks) + g + xnb - yn

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

            mom0 = 1. - (1. - theta)*yc/(w*nc)
            mom1 = 1. - Ecs/Eys
            mom2 = 1. - (tax_rev - tran - netb)/g
            mom3 = 1. - (Ecc  + Ex+ (grate + delk)*(kc + Eks) + g + xnb - yn)/yc
            print('')

            print('1-(1-thet)*yc/(E[w*eps*n]) = {}'.format(mom0))
            print('1-E(cs)/E(ys) = {}'.format(mom1))
            #print('1-((1-taud)kc+E(ks)+b)/Ea = {}'.format(1. - (b + (1.- taud)*kc + Eks)/Ea))
            print('1-(tax-tran-netb)/g = {}'.format(mom2))
            print('1-(Ecc+Ex+(grate+delk)*(kc + Eks)+ g + xnb - yn)/yc = {}'.format(mom3))

            print('')
            print('Important Moments')
            print('  Financial assets = {}'.format(Ea))
            print('  Sweat assets = {}'.format(Ekap))
            print('  Govt. debt = {}'.format(b))
            print('  S-Corp Rental Capital (ks) =  {}'.format(Eks))
            print('  Ratio of C-corp workers = {}'.format(EIc))
            print('  GDP = {}'.format(GDP))

            print('')
            print('C-corporation production:')
            print('  Consumption = {}'.format(Ecc))
            print('  Investment = {}'.format(kc*(grate + delk)))
            print('  Govt purchase = {}'.format(g))
            print('  Output = {}'.format(yc))
            print('  Capital (kc) = {}'.format(kc))
            print('  Effective Worked Hours = {}'.format(En)) 

            print('')
            print('S-corporation production:')
            print('  Consumption = {}'.format(Ecs))
            print('  Output = {}'.format(Eys))
            print('  investment, sweat = {}'.format(Ex))
            print('  Hours, production = {}'.format(Emy))
            print('  Hours, sweat = {}'.format(Emx))
            print('  Sweat equity (kap) = {}'.format(Ekap))

            print('')
            print('National Income Shares (./GDP):')
            print('  C-wages (w*nc) = {}'.format((w*nc)/GDP))
            print('  Rents = {}'.format((rc*kc + rs*Eks)/GDP))
            print('  Sweat = {}'.format((p*Eys - (rs+delk)*Eks)/GDP))
            print('  Deprec. = {}'.format((delk*(kc+Eks))/GDP))
            print('  NonBusiness income = {}'.format(yn/GDP))


            print('')
            print('National Product Shares (./GDP):')
            print('  Consumption = {}'.format(C/GDP))
            print('  Investments(xc) = {}'.format((xc)/GDP))
            print('  Investments(xs) = {}'.format((Exs)/GDP))
            print('  Govt Consumption = {}'.format(g/GDP))
            print('  Nonbusiness investment = {}'.format((xnb)/GDP))

            print('')
            print('Govt Budget:')
            print('  Public Consumption = {}'.format(g))
            print('  Net borrowing = {}'.format(netb))
            print('  Transfer = {}'.format(tran))
            print('  Tax revenue = {}'.format(tax_rev))

            print('')
            print('Gini Coefficients:')
            print('  Financial Assets = {}'.format(gini(data_ss[:,1])))
            print('  Sweats Assets = {}'.format(gini(data_ss[:,2])))
            print('  C-wages = {}'.format(gini(w*data_ss[:,5]* data_ss[:,10]*data_ss[:,0])))
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
                print(' N = ', t, ', with = {:f}'.format((np.mean(np.all(data_is_c[:,-(t+2):-1] == False, axis = 1)) -                      np.mean(np.all(data_is_c[:,-(t+3):-1] == False, axis = 1)) )/ np.mean(1. - data_ss[:,0]))) 
        #     print(np.mean(data_is_c[:,-21:-1] == False)/np.mean(1. - data_ss[:,0]))


        mom0 = comm.bcast(mom0)
        mom1 = comm.bcast(mom1)
        mom2 = comm.bcast(mom2)
        mom3 = comm.bcast(mom3)

        self.moms = [mom0, mom1, mom2, mom3]

        return

    def calc_sweat_eq_value(self):
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

        agrid = Econ.agrid
        kapgrid = Econ.kapgrid
        epsgrid = Econ.epsgrid
        zgrid = Econ.zgrid

        prob = Econ.prob

        is_to_iz = Econ.is_to_iz
        is_to_ieps = Econ.is_to_ieps

        amin = Econ.amin
        num_suba_inner = Econ.num_suba_inner
        num_subkap_inner = Econ.num_subkap_inner

        num_a = Econ.num_a
        num_kap = Econ.num_kap
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
        xi9 = Econ.xi9
        xi10 = Econ.xi10
        xi11 = Econ.xi11
        xi12 = Econ.xi12

        #load the value functions
        vc_an = Econ.vc_an
        vs_an = Econ.vs_an
        vs_kapn = Econ.vs_kapn
        vcn = Econ.vcn
        vsn = Econ.vsn  

        get_cstatic = Econ.generate_cstatic()
        get_sstatic = Econ.generate_sstatic()
        dc_util = Econ.generate_dc_util()

        #obtain the value function and the discounted expected value function
        vn = np.fmax(vcn, vsn) #the value function
        EV = bh*((vn**(1. - mu))@(prob.T)) # the discounted expected value function


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
                for ikap, kap in enumerate(kapgrid):
                    for istate in range(num_s):

                        an = None
                        kapn = None

                        an_s = vs_an[ia, ikap, istate]
                        kapn_s = vs_kapn[ia, ikap, istate]
                        z = zgrid[is_to_iz[istate]]

                        an_c = vc_an[ia, ikap, istate]
                        kapn_c = la * kap
                        eps = epsgrid[is_to_ieps[istate]]

                        ####this does not work actually... infeasible points are possibly chosen due to inter/extra-polation
                        #val_s = fem2d_peval(an_s, kapn_s, agrid, kapgrid, vsn[:,:, istate])
                        #val_c = fem2d_peval(an_c, kapn_c, agrid, kapgrid, vcn[:,:, istate])

                        val_c = (get_cstatic([a, an_c, eps])[0]  + fem2d_peval(an_c, kapn_c, agrid, kapgrid, EV[:,:,istate])) **(1./(1.- mu))
                        val_s = (get_sstatic([a, an_s, kap, kapn_s, z])[0]  + fem2d_peval(an_s, kapn_s, agrid, kapgrid, EV[:,:,istate])) **(1./(1.- mu))


                        if val_s >= val_c:
                            to_be_s[ia, ikap, istate] = True

                            an = an_s
                            kapn = kapn_s


                            u, cc, cs, cagg, l, mx, my, x, ks, ys = get_sstatic([a, an, kap, kapn, z])


                            #print(f'u = {u}')
                            u_c[ia, ikap, istate] = dc_util(cagg, l)

                            profit = p*ys - (rs + delk)*ks - x #this can be nagative
                            tax = taum * max(profit, 0.)
                            div = phi * p * ys - x #div related to sweat equity


                            d[ia, ikap, istate] = div
    #                         d_after_tax[ia, ikap, istate] = div - tax


                        else:
                            to_be_s[ia, ikap, istate] = False
                            d[ia, ikap, istate] = 0.0
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

                                u, cc, cs, cagg, l, mx, my, x, ks, ys = get_sstatic([an, anp, kapn, kapnp, zp])
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
        def _inner_get_sweat_equity_value_pp_(_d_, _val_):
            bEval = np.zeros((num_a, num_kap, num_s))

            for ia in nb.prange(num_a):
                a = agrid[ia]
                for ikap in range(num_kap):
                    kap = kapgrid[ikap]
                    for istate in range(num_s):
                        dc_u = u_c[ia, ikap, istate]

                        for istate_n in range(num_s):

        #                     an = an2[ia, ikap, istate, istate_n]
        #                     kapn = kapn2[ia, ikap, istate, istate_n]

                            an = an1[ia, ikap, istate]
                            kapn = kapn1[ia, ikap, istate]

                            dc_un = up_c[ia, ikap, istate, istate_n]
                            val_p = fem2d_peval(an, kapn, agrid, kapgrid, _val_[:, :, istate_n])

            #                     val_p[ia, ikap, istate, istate_n] = fem2d_peval(an, kapn, agrid, kapgrid, val[:, :, istate_n])

                            bEval[ia, ikap, istate] += prob[istate, istate_n] * dc_un * val_p

                        #after taking expectation
                        bEval[ia, ikap, istate] =  bh * bEval[ia, ikap, istate] / dc_u

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
            val_tmp[:] = _inner_get_sweat_equity_value_pp_(d, val)


            dist = np.max(np.abs(val_tmp - val))

            if dist < tol:
                break


            val[:] = val_tmp[:]

        t1 = time.time()

        if it >= maxit:
            print('Warning: iteration reached the max iteration.')
        print(f'elapsed time = {t1 - t0}')
        print(f'{it+1}th loop. dist = {dist}')

        self.sweat_div = d
        self.sweat_val = val
        
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
        zeta= Econ.zeta

        agrid = Econ.agrid
        kapgrid = Econ.kapgrid
        epsgrid = Econ.epsgrid
        zgrid = Econ.zgrid

        prob = Econ.prob

        is_to_iz = Econ.is_to_iz
        is_to_ieps = Econ.is_to_ieps

        amin = Econ.amin
        num_suba_inner = Econ.num_suba_inner
        num_subkap_inner = Econ.num_subkap_inner

        num_a = Econ.num_a
        num_kap = Econ.num_kap
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
        xi9 = Econ.xi9
        xi10 = Econ.xi10
        xi11 = Econ.xi11
        xi12 = Econ.xi12

        data_a = Econ.data_a
        data_kap = Econ.data_kap
        data_i_s = Econ.data_i_s
        data_is_c = Econ.data_is_c

        #simulation parameters
        sim_time = Econ.sim_time
        num_total_pop = Econ.num_total_pop

        get_cstatic = Econ.generate_cstatic()
        get_sstatic = Econ.generate_sstatic()


        @nb.jit(nopython = True, parallel = True)
        def calc_all(data_a_, data_kap_, data_i_s_, data_is_c_,
                    data_u_, data_cc_, data_cs_, data_cagg_, data_l_, data_n_, data_mx_, data_my_, data_x_, data_ks_, data_ys_):

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
                    mx = np.nan
                    my = np.nan
                    ks = np.nan
                    ys = np.nan

                    a = data_a_[i, t-1]
                    kap = data_kap_[i, t-1]

                    an = data_a_[i, t]
                    kapn = data_kap_[i, t]

                    is_c = data_is_c_[i, t]

                    if is_c:
                        u, cc, cs, cagg, l ,n = get_cstatic([a, an, eps])
                    else:
                        u, cc, cs, cagg, l, mx, my, x, ks, ys = get_sstatic([a, an, kap, kapn, z])

                    data_u_[i, t] = u
                    data_cc_[i, t] = cc
                    data_cs_[i, t] = cs
                    data_cagg_[i, t] = cagg
                    data_l_[i, t] = l
                    data_n_[i, t] = n
                    data_mx_[i, t] = mx
                    data_my_[i, t] = my
                    data_x_[i, t] = x
                    data_ks_[i, t] = ks
                    data_ys_[i, t] = ys
                    
        data_u = np.zeros(data_a.shape)
        data_cc = np.zeros(data_a.shape)
        data_cs = np.zeros(data_a.shape)
        data_cagg = np.zeros(data_a.shape)
        data_l = np.zeros(data_a.shape)
        data_n = np.zeros(data_a.shape)
        data_mx = np.zeros(data_a.shape)
        data_my = np.zeros(data_a.shape)
        data_x = np.zeros(data_a.shape)
        data_ks = np.zeros(data_a.shape)
        data_ys = np.zeros(data_a.shape)
        data_ys = np.zeros(data_a.shape)

        #note that this does not store some impolied values,,,, say div or value of sweat equity
        calc_all(data_a, data_kap, data_i_s, data_is_c, ##input
             data_u, data_cc, data_cs, data_cagg, data_l, data_n, data_mx, data_my, data_x, data_ks, data_ys ##output
            )

        data_val_sweat = np.zeros(data_a.shape)
        data_div_sweat = np.zeros(data_a.shape)



        @nb.jit(nopython = True, parallel = True)
        def calc_val_seq(data_a_, data_kap_, data_i_s_, data_is_c_, sweat_eq_val_ ,data_val_seq_):
        
            for t in nb.prange(1, sim_time):
                for i in range(num_total_pop):
            
                    istate = data_i_s_[i, t]
                    eps = epsgrid[is_to_ieps[istate]]
                    z = zgrid[is_to_iz[istate]]
                
                    a = data_a_[i, t-1]
                    kap = data_kap_[i, t-1]
            
                    an = data_a_[i, t]
                    kapn = data_kap_[i, t]
                
                #             is_c = data_is_c_[i, t]
                
                    data_val_seq_[i,t] = fem2d_peval(a, kap, agrid, kapgrid, sweat_eq_val_[:,:,istate])
                
        sweat_div = self.sweat_div
        sweat_val = self.sweat_val

        calc_val_seq(data_a, data_kap, data_i_s, data_is_c, sweat_div, data_div_sweat)
        calc_val_seq(data_a, data_kap, data_i_s, data_is_c, sweat_val, data_val_sweat)


        self.data_u = data_u
        self.data_cc = data_cc
        self.data_cs = data_cs
        self.data_cagg = data_cagg
        self.data_l = data_l
        self.data_n = data_n
        self.data_mx = data_mx
        self.data_my = data_my
        self.data_x = data_x
        self.data_ks = data_ks
        self.data_ys = data_ys
        self.data_val_sweat = data_val_sweat
        self.data_div_sweat = data_div_sweat


        return
    
    def save_result(self, dir_path_save = './save_data/'):
        if rank == 0:
            print('Saving results under ', dir_path_save, '...')


            
            np.save(dir_path_save + 'data_a', self.data_a[:, -100:])
            np.save(dir_path_save + 'data_kap', self.data_kap[:, -100:])
            np.save(dir_path_save + 'data_i_s', self.data_i_s[:, -100:])
            np.save(dir_path_save + 'data_is_c', self.data_is_c[:, -100:])
            np.save(dir_path_save + 'data_u', self.data_u[:, -100:])
            np.save(dir_path_save + 'data_cc', self.data_cc[:, -100:])
            np.save(dir_path_save + 'data_cs', self.data_cs[:, -100:])
            np.save(dir_path_save + 'data_cagg', self.data_cagg[:, -100:])
            np.save(dir_path_save + 'data_l', self.data_l[:, -100:])
            np.save(dir_path_save + 'data_n', self.data_n[:, -100:])
            np.save(dir_path_save + 'data_mx', self.data_mx[:, -100:])
            np.save(dir_path_save + 'data_my', self.data_my[:, -100:])
            np.save(dir_path_save + 'data_x', self.data_x[:, -100:])
            np.save(dir_path_save + 'data_ks', self.data_ks[:, -100:])
            np.save(dir_path_save + 'data_ys', self.data_ys[:, -100:])
            np.save(dir_path_save + 'data_ss', self.data_ss)

            np.save(dir_path_save + 'vc_an', self.vc_an)
            np.save(dir_path_save + 'vs_an', self.vs_an)
            np.save(dir_path_save + 'vs_kapn', self.vs_kapn)
            np.save(dir_path_save + 'vcn', self.vcn)
            np.save(dir_path_save + 'vsn', self.vsn)

            np.save(dir_path_save + 'sweat_div', self.sweat_div)
            np.save(dir_path_save + 'sweat_val', self.sweat_val)

            np.save(dir_path_save + 'data_div_sweat', self.data_div_sweat[:, -100:])
            np.save(dir_path_save + 'data_val_sweat', self.data_val_sweat[:, -100:])

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


if __name__ == '__main__':
    econ = import_econ()

    econ.get_policy()
    econ.simulate_model()

    export_econ(econ)
