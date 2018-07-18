from SCEconomy_lifecycle_give_A import *

from mpi4py import MPI
comm = MPI.COMM_WORLD #retreive the communicator module
rank = comm.Get_rank() #get the rank of the process
size = comm.Get_size() #get the number of processes

e = import_econ()
#e.sim_time = 300
e.simulate_model()
export_econ(e)



comm.Barrier()
if rank == 0:
    import os
    os.system('say "your program has finished"')
