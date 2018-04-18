from SCEconomy_lifecycle_give_A import *
e = import_econ()
#e.sim_time = 300
e.simulate_model()
export_econ(e)

#if rank == 0:
import os
os.system('say "your program has finished"')
