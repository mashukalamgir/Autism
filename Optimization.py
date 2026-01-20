import numpy as np
import math
import scipy.stats

def opt():
    lb = -0.1
    ub = 0.1
    r1 = np.random.uniform(0, 1)
    r2 = np.random.uniform(0, 1)
    InitialPop = np.random.uniform(lb, ub, size=(5, 5))
    
    def fitness_computation(popdata):
        for i in popdata:
            fitdata = 0
            for j in popdata:
                fitdata = fitdata+(j*j)
        return fitdata
    f = fitness_computation(InitialPop)
    bf = min(f)
    hgso = np.zeros((5, 5))
    
    for i in range(len(InitialPop)):
        for j in range(len(InitialPop[0])):
            rand = np.random.uniform(0, 1)
            r3 = np.random.uniform(0, 1)
            r4 = np.random.uniform(0, 1)
            r5 = np.random.uniform(0, 1)
            a = 2 * (1 - (i / len(InitialPop)))
            Rvec = 2 * a * rand - a
            sech = 2 / (math.exp(i) +  math.exp(-i))
            E = sech * (f[i] - bf)
            l = scipy.stats.norm(0, 1).pdf(0)
            Shungry = sum(sum(InitialPop))
            if r3 < l:
                W1 = InitialPop[i][j] * (len(InitialPop)/Shungry) * r4
            else:
                W1 = 1
            W2 = (1 - math.exp(-InitialPop[i][j] - Shungry)) * r5 * 2
            if  r1 < l:
                hgso[i][j] = InitialPop[i][j] * (1 - l)
            elif r1 > l or r2 > E:
                hgso[i][j] = W1 * bf + Rvec * W2 * bf - InitialPop[i][j]
            elif r1 > l or r2 < E:
                hgso[i][j] = W1 * bf - Rvec * W2 * bf - InitialPop[i][j]
        
    h =  fitness_computation(hgso)
    
    WF = max(h)
    BF = min(h)
    
    TH = []
    for i in range(len(h)):
        r6 = np.random.uniform(0, 1)
        th = (h[i] - BF) / (WF - BF) * r6 * 2 * (ub-lb)
        TH.append(th)
        
    TH = np.array(TH)
    LH = -1
    
    H = []
    for i in range(len(h)):
        if TH[i] < LH:
            r = np.random.uniform(0, 1)
            h1 = LH * (1 + r)
            H.append(h1)
            
        elif TH[i] >= LH:
            H.append(TH[i])
    
    hungry = []   
     
    for i in range(len(h)):
        if h[i] == BF:
            hungry.append(0)
        else:
            hungry.append(H[i])
    hungry.sort()
    return hungry