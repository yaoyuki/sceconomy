from SCEconomy_lifecycle_give_A import *

from mpi4py import MPI
comm = MPI.COMM_WORLD #retreive the communicator module
rank = comm.Get_rank() #get the rank of the process
size = comm.Get_size() #get the number of processes


def curvedspace(begin, end, curve, num=100):
    import numpy as np
    ans = np.linspace(0, (end - begin)**(1.0/curve), num) ** (curve) + begin
    ans[-1] = end #so that the last element is exactly end
    return ans



agrid2 = curvedspace(0., 100., 2., 40)
# kapgrid2 = curvedspace(0., 1., 2., 20)
zgrid2 = np.load('./input_data/zgrid.npy') ** 2.
# prob2 = np.load('./input_data/transition_matrix.npy')
e = Economy(agrid = agrid2, zgrid = zgrid2)
e.set_prices(w = 3.1480348355657055,
             p = 1.0207889942205437,
             rc = 0.06202604135907751)
e.get_policy()
export_econ(e)

import os
os.system('say "your program has finished"')
