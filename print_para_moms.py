from SCEconomy_give_A import *

e = import_econ()

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


