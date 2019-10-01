from quspin.operators import hamiltonian, exp_op # Hamiltonians and operators
import numpy as np # generic math functions
from quspin.basis import spinless_fermion_basis_1d # Hilbert space spinless fermion basis
from numpy.random import uniform,choice # tools for doing random sampling
#from quspin.tools.measurements import ent_entropy, diag_ensemble # entropies
from math import log
import random
from timeit import default_timer as timer 
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

start = timer() ## keep track of running time

n=2 

def ham(L, J, t1, t2): #define the 1-D dimerized random hopping model with n-flavors of Majorana fermions
    sigmaj=J/(n**(1.5)) # variance of interaction
    sigmat1=t1/(n**(0.5)) # variance of odd bond hopping
    sigmat2=t2/(n**(0.5)) # variance of even bond hopping
    hop1=0.5*np.random.normal(0, sigmat1, (L,n,n)) # the factor 0.5 and 0.25 is in accordance with the definition in MMA code
    hop2=0.5*np.random.normal(0, sigmat2, (L,n,n))
    intj=0.25*np.random.normal(0, sigmaj, L)
    
    t1_pp=[[1j*hop1[i,0,0]-1j*hop1[i,1,1]+2*hop1[i,0,1],i,(i+1)%L] for i in range(0,L,2)]
    t1_mm=[[1j*hop1[i,0,0]-1j*hop1[i,1,1]-2*hop1[i,0,1],i,(i+1)%L] for i in range(0,L,2)]
    t1_mp=[[1j*hop1[i,0,0]+1j*hop1[i,1,1],i,(i+1)%L] for i in range(0,L,2)]
    t1_pm=[[1j*hop1[i,0,0]+1j*hop1[i,1,1],i,(i+1)%L] for i in range(0,L,2)]

    t2_pp=[[1j*hop2[i,0,0]-1j*hop2[i,1,1]+2*hop2[i,0,1],i,(i+1)%L] for i in range(1,L,2)]
    t2_mm=[[1j*hop2[i,0,0]-1j*hop2[i,1,1]-2*hop2[i,0,1],i,(i+1)%L] for i in range(1,L,2)]
    t2_mp=[[1j*hop2[i,0,0]+1j*hop2[i,1,1],i,(i+1)%L] for i in range(1,L,2)]
    t2_pm=[[1j*hop2[i,0,0]+1j*hop2[i,1,1],i,(i+1)%L] for i in range(1,L,2)]
    
    J_int=[[-4*intj[i],i,(i+1)%L] for i in range(L)]    

    basis = spinless_fermion_basis_1d(L=L,Nf=range(0,L+1,2)) #even number sector
    static =[["+-",t1_pm],["-+",t1_mp],["++",t1_pp],["--",t1_mm],["+-",t2_pm],["-+",t2_mp],["++",t2_pp],["--",t2_mm],["zz",J_int]]
    dynamic=[] # time-dependent part of H
    no_checks={"check_herm":False,"check_pcon":False,"check_symm":False} 
    H=hamiltonian(static,dynamic,dtype=np.complex128,basis=basis,**no_checks)   
    return H

def level(e): #calculate the energy level statistics
    lev=np.zeros(len(e))
    for i in range(len(e)-1):
        lev[i]=e[i+1]-e[i]
    cond=lev>10**-6 # removing the degeneracies
    lev=lev[cond]
    ratio=np.zeros(len(lev)-1)
    for i in range(len(ratio)):
        s1=lev[i+1]
        s2=lev[i]
        ratio[i]=min(s1,s2)/max(s1,s2)
    return np.mean(ratio)

#entanglement spectrum for 80% of the total eigenstates
def ev(h):
    h_dense=h.todense() #convert h into dense matrix for fully diagonalization
    en, wave = eigh(h_dense)
    p=int(0.1*2**(L-1))
    wav = wave[p:-p] #discard the largest and smallest 10% eigenstates to improve accuray, see RPB 95, 245134 for details
    es = np.zeros((len(wav),int(2**(L//2+1))), dtype='float64')
    basis = spinless_fermion_basis_1d(L=L,Nf=range(0,L+1,2)) #even number sector   
    #calculate the half-chain entanglement spectrum for all selected eigenstates
    for i in range(len(wav)):
        ps=wav[i,:]
        S = basis.ent_entropy(ps,sub_sys_A=tuple(range(L//2+1)), density=False, return_rdm="A")
        rdm_A = S["rdm_A"]
        es[i] = -eigh(np.log(rdm_A),eigvals_only=True) #entanglement spectrum
    ind = len(wave)//2   
    stat = level(en) # calculate the energy level statistics
    #only the 200 middle states will be kept
    return es, en, stat, wave[ind-100:ind+100,:]


# read model parameters
para = open('para_model.txt', 'r')
para = para.readlines()
L, j, t1, t2 = int(para[0]), float(para[1]), float(para[2]), float(para[3]) 

# generate Hamiltonian and running the calculation
h = ham(L,j,t1,t2)
result=ev(h)
# generate random bytes to name the output file
# it's a workaround for unability to read the current file name
x=np.random.bytes(12)
# save the output into compressed npz file
np.savez_compressed('%s_en_spectrum_L=%s_j=%s'%(x,L,j), ent=result[0], en=result[1], levelstat=result[2], wave=result[3])
end = timer()
print("Elapsed = %s" % (end - start))