import numpy as np
import matplotlib.pyplot as plt

H_0=0.3
N=1000
sides=np.random.choice([True, False], size=N, p=[H_0, 1-H_0])        # generates 1000 tosses and assignes value True to heads

# I need to define the prior, likelyhood and the posterior
####################
def prior(pH):
    P=np.zeros_like(pH)
    P[(0<=pH) & (pH<=1)]=1
    return P
####################
def likelyhood(data, pH):
    N=len(data)
    R=sum(data)
    return pH**R*(1-pH)**(N-R)
####################

# def posterior(data, pH):
#     p=likelyhood(data, pH)*prior(pH)
#     p/= np.trapz(p, pH)
    
#     return p
def posterior(pH,data):
    p=likelyhood(data, pH)*prior(pH)
    C=np.max(p)
    #C=np.sum(p)
    return p/C
####################

####################
# I repeat the process for a gaussian prior centered at 0.5 with SD=1
def Gprior(pH):
    g=np.zeros_like(pH)
    sd=0.1
    for i in range(len(g)):
        g[i]=1/(4*sd*np.sqrt(2*np.pi))*np.exp(-1/2*((pH[i]-0.5)/sd)**2)
    return g 

def Gposterior(pH, data):
    p=likelyhood(data, pH)*Gprior(pH)
    C=np.max(p)
    #C=np.sum(p)
    return p/C
        
pH=np.linspace(0, 1, N)

fig, axs=plt.subplots(nrows=3, ncols=3, sharex=True, sharey=False, figsize=(10, 10))
vaxs=np.reshape(axs,-1)
vaxs[0].plot(pH,prior(pH))
vaxs[0].plot(pH,Gprior(pH))
M=[1, 50, 100, 300, 700, 1000]
for i in range(len(M)):
    ax=vaxs[i+1]
    ax.plot(pH, posterior(pH, sides[:M[i]] ))
    ax.plot(pH, Gposterior(pH, sides[:M[i]] ))
    ax.text(0.7, 0.9, '$N={0}$'.format(M[i]))
    
for row in range(3): axs[row,0].set_ylabel('$p(p_H|D_\mathrm{obs},I)$')
for col in range(3): axs[-1,col].set_xlabel('$p_H$')