from SCEconomy_bizmkt_give_A import *

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
zgrid2 = np.load('./input_data/zgrid.npy') ** 2.
e = Economy(agrid = agrid2, zgrid = zgrid2)
e.set_prices(w = 3.123668659426386,
             p = 0.995310563254898,
             rc = 0.06327936800968331,
             pkap = 1.0, #need a good initial guess. later we can always use (6)
             kapbar = 1.0)

e.get_policy()
export_econ(e)

comm.Barrier()
if rank == 0:
    import os
    os.system('say "your program has finished"')
