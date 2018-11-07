import numpy as np
from SCEconomy_s_emp import *

e = import_econ()

np.save('./save_data/data_i_s', e.data_i_s[:, -100:])
np.save('./save_data/agrid', e.agrid)
np.save('./save_data/kapgrid', e.kapgrid)
np.save('./save_data/epsgrid', e.epsgrid)
np.save('./save_data/zgrid', e.zgrid)
np.save('./save_data/prob', e.prob)
np.save('./save_data/is_to_iz', e.is_to_iz)
np.save('./save_data/is_to_ieps', e.is_to_ieps)


print('agrid')
print(e.agrid)

print('kapgrid')
print(e.kapgrid)

print('zgrid')
print(e.zgrid)

print('epsgrid')
print(e.epsgrid)

print('prob')
print(e.prob)

e.print_parameters()
e.calc_moments()



