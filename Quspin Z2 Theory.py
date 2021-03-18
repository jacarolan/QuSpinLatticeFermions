from __future__ import print_function, division
import sys,os
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from quspin.basis import spinless_fermion_basis_1d # fermion basis for comparison
import numpy as np # generic math functions
import matplotlib.pyplot as plt
import itertools
plt.rcParams['figure.figsize'] = [10, 8] # changes default figure size in jupyter notebook


# ## Coupling and option initialization

# In[2]:


N=9 # number of sites (NOT STAGGERED SITES!!)
Pinit=1 # P_{-1}
g = -1 # E-field coupling constant
t = -1 # coupling constant for hopping term
m = -1 # mass

half_filling = 1 # options: 1 for half filling, 0 for not half-filling
single_filling = 0 # options: 1 for single-filling, 0 else

PBC = 1 # options: 1 for periodic boundary conditions, 0 for fixed

if(N % 4 < 2): # Checks if N is even for anti- or periodic boundary conditions
    Neven = 1
else:
    Neven = 0


# In[3]:


def arrayn(n): ## returns an array of length n with values 0,...,n-1
    return([i for i in range(n)])

if(half_filling): # sets filling to N//2 if half filling is turned on, else it does not specify
    basis1 = spin_basis_1d(N,pauli=-1,Nup=N//2)
elif(single_filling):
    basis1 = spin_basis_1d(N,pauli=-1,Nup=1)
else:
    basis1 = spin_basis_1d(N,pauli=-1)


# In[4]:


###### Hopping terms
if(PBC):
    
    if(Neven and half_filling): # Anti-Periodic
        J12 = [[-t,i,(i+1)%N] for i in range(N-1)] # The final boundary term must be added with negative coupling
        J13 = [[-t,(i+1)%N,i] for i in range(N-1)] 
        J12.append([t,N-1,0]) # The negatively-coupled (anti-periodic) final boundary terms
        J13.append([t,0,N-1])
    
    else: # Periodic 
        J12 = [[-t,i,(i+1)%N] for i in range(N)] # the %N and range increased by 1 adds the final boundary term making it periodic
        J13=[[-t,(i+1)%N,i] for i in range(N)]
    
else: # Fixed boundary conditions
    J12 = [[-t,i,i+1] for i in range(N-1)] # we need to append "+-" for signs of sigmas
    J13=[[-t,i+1,i] for i in range(N-1)] # append "+-" for signs of sigmas, i+1,i since this is h.c. term

###### mass term (unchanged by PBC)
J14 = [[m*(-1.0)**i,i,i] for i in range(N)] # append  "+-" for signs of sigmas


# In[5]:


staticpart1=[["+-",J14],["+-",J12],["+-",J13]] # Defining static and dynamic parts of H and filling with hopping and mass terms
dynamicpart1=[]


# In[6]:


###### E-field interaction term

zstring = "z"
def upperindices(n): # Creating a function to make the right length string of "z"
    a = ""
    for i in range(n):
        a = a + zstring
    return(a)

def summandJ1(n): # Creating a function to generate coupling lists for the E-field product term
    indices=[i for i in range(n+1)]
    indices.insert(0,-Pinit*g*(-1.0)**(n*(n+1.0)*.5)*(-1.0)**(n+1.0))
    indices=[upperindices(n+1),[indices]]
    return(indices)

for i in range(0,N): # Piecemeal inserting the E-field product term into the hamiltonian
        staticpart1.insert(0,summandJ1(int(i)))


# In[7]:


H1 = hamiltonian(staticpart1,dynamicpart1,basis=basis1,dtype=np.float64) # Creating the hamiltonian with proper basis

eigenvalues1 = H1.eigvalsh() # Getting and plotting eigenvalues
plt.plot(eigenvalues1,'+')

print("Static couplings spin:", staticpart1)

print("# of eigenvalues: " + str(len(eigenvalues1))) # Printing number of eigenvalues for comparison
#x=np.linspace(0,50,500)
#plt.plot(x,np.sqrt(1+np.sin(2*np.pi*(x-25.0)/100.0)**2))


# Note: Disagrees with Niklas's code for PBC = 1 and N is EVEN

# ## Do it again with Fermion basis to test

# In[8]:


# Define basis
if(half_filling): # sets filling to N//2 if half filling is turned on, else it does not specify
    basis = spinless_fermion_basis_1d (L=N,Nf=N//2)
elif(single_filling):
    basis = spinless_fermion_basis_1d (L=N,Nf=1)
else:
    basis = spinless_fermion_basis_1d (L=N)

###### Hopping terms
if(PBC):
    
    J2 = [[-t,i,(i+1)%N] for i in range(N)] # fermionic basis in quspin gets rid of need for manual anti-periodicity adjustment
    J3=[[-t,(i+1)%N,i] for i in range(N)]
    
else: # Fixed boundary conditions
    J2 = [[-t,i,i+1] for i in range(N-1)] # we need to append "+-" for daggers of psis
    J3=[[-t,i+1,i] for i in range(N-1)] # append "+-" for daggers of psis, i+1,i since this is h.c. term

###### mass term (unchanged by PBC)
J4 = [[m*(-1.0)**i,i,i] for i in range(N)] # append  "+-" for daggers of psis


# In[9]:


staticpart=[["+-",J4],["+-",J2],["+-",J3]] # Defining static and dynamic parts of H and filling with hopping and mass terms
dynamicpart=[]


# In[10]:


###### E-field interaction term

zstring = "z"
def upperindices(n): # Creating a function to make the right length string of "z"
    a = ""
    for i in range(n):
        a = a + zstring
    return(a)

def summandJ(n): # Creating a function to generate coupling lists for the E-field product term
    indices=[i for i in range(n+1)]
    indices.insert(0,-Pinit*g*(-1.0)**(n*(n+1.0)*.5)*(-2.0)**(n+1.0)) # The extra -2^n is due to the fermion basis "z" defn.
    indices=[upperindices(n+1),[indices]]
    return(indices)

for i in range(N): # Piecemeal inserting the E-field product term into the hamiltonian
        staticpart.insert(0,summandJ(int(i)))


# In[11]:


H = hamiltonian(staticpart,dynamicpart,basis=basis,dtype=np.float64) # Creating the hamiltonian with proper basis

eigenvalues = H.eigvalsh() # Getting and plotting eigenvalues0
f = plt.figure(figsize=(8,8))

print("Static couplings fermion:", staticpart)

ax = plt.subplot(111)

ax.plot(eigenvalues,'o')
ax.plot(eigenvalues1,'+')


print("# of eigenvalues: " + str(len(eigenvalues))) # Printing number of eigenvalues for comparison


# Note: Agreeing! :) 

# In[12]:


print("Eigenvalues:\n", eigenvalues1)

