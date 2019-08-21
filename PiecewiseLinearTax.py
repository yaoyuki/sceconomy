import numpy as np

def get_consistent_phi(bras, taus, psi, j):
    
    #inputs
    
    num = len(taus)
        
    if len(bras) + 1 != num:
        print('error: len(bras) != len(taus)')
        
        return None
    
    if j < 0 or j > num-1:
        print('j does not fall in the range')
    
        return None
    
    psis = np.zeros(num)
    psis[j] = psi
    
    for i in range(j+1, num):
        psis[i] = -((taus[i-1] - taus[i])*bras[i-1] - psis[i-1])
        
    for i in reversed(range(0, j)):
        psis[i] = (taus[i] - taus[i+1])*bras[i] + psis[i+1]    
        
    return psis
