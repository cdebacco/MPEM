'''
Functions needed to manipulate the observables (e.g. magnetizations, marginals, etc...)
'''

import numpy as np
from copy import copy,deepcopy
import math 

def joint_different_t(u,v,G,d):
    " Calcualtes joint probability of two neighboring nodes, one at t and the other at t-1"
    joint=np.zeros((d,d));  # Joint probability P_ij(s_v(t),s_u(t-1))
    A=G[u][v]['As'];
    B=G[v][u]['As'];
    t=len(A)-2;
    # Build K[0]
    a=np.einsum('jab->jba',A[0]); # Transpose A[0]
    K=np.einsum('iab,jbc->ijac',B[0],a); # Matrix product
    for s in range(t-1):
        L=np.einsum('jiab,sjbc->isac',B[s+1],K);
        a=np.einsum('ijab->jiba',A[s+1]); # Transpose A[s+1]
        K=np.einsum('isab,jsbc->ijac',L,a);
        norm=np.linalg.norm(K); # Normalize with the Frobenius or 2-norm        
    # step t-1
    L=np.einsum('jiab,sjbc->isac',B[t],K);
    a=np.einsum('ijab->jiba',A[t]); # Transpose A[t](s_i(t-1))
    K=np.einsum('isab,jsbc->ijsac',L,a);    # DO NOT sum over s_i(t-1) !!!!!
    # Last time step
    a=np.einsum('ijab->ijba',A[t+1]); # Transpose A[t+1]
    kA= np.einsum('ijsab,imbc->jsac',K,a);
    K=np.einsum('jlab,jsbc->jsac',B[t+1],kA);   # ATTENTION: K(s_j(t), s_i(t-1))   !!!!
    K=K.reshape(d,d);
    # Build joint P_ij(s_j(t), s_i(t-1)) -----------------------
    joint=K/np.einsum('ji->',K); # Normalize        
    return joint;


def calulate_marginals(G,d,out_mag,out_cor):
    " For a given time step, use the A's to calculate the marginal P_i(s_i(t+1))"
    " Store it into G.node[u]['marginal'][t] "
    Correlation=0.;count=0;
    
    for u in G.nodes():
        m_u=G.node[u]['marginal'][1]-G.node[u]['marginal'][0]; # m_i(t-1)
        G.node[u]['marginal']=np.ones((d,));
        for n in range(G.out_degree(u)):
            v=G.neighbors(u)[n] ;  # Pick the first neighbor
            joint=joint_different_t(v,u,G,d);  # Joint probability P_ij(s_u(t),s_v(t-1)) --> pay attention to the right order u,v

            # Calculate correlation -----------------
            C_ij=(joint[0,0]+joint[1,1])-(joint[0,1]+joint[1,0]);    # < s_i(t) s_j(t-1) >
            X=np.einsum('ij->i',joint); # Marginalize over  j   -> P(s_i(t))
            Y=np.einsum('ij->j',joint); # Marginalize over i    -> P(s_j(t-1))
            C_ij-=(-X[0]+X[1])*(-Y[0]+Y[1]);    #  - <s_i(t)><s_j(t-1)>
            Correlation+=(C_ij)#*G[u][v]['J'];
            print >> out_cor, count,u,v, C_ij;
            count+=1;
            
            if(n==0):   # Only need one neighbor for the marginal
                G.node[u]['marginal']=X.copy();
                m_u=(G.node[u]['marginal'][1]-G.node[u]['marginal'][0])-m_u;
                m_u*=m_u;
                print >> out_mag, u,(G.node[u]['marginal'][1]-G.node[u]['marginal'][0]);
    
    Correlation/=float(G.number_of_edges());
    return Correlation;  
 
def calulate_observable(G,d):
    " For a given time step, use the marginals to calculate the average observables"
    m=0.; q=0.;
    z=float(G.number_of_nodes());
    for u in G.nodes():
        M=G.node[u]['marginal'][1]-G.node[u]['marginal'][0];
        m+=M/z;
        q+=M*M/z;
    return m,q;     

def max_dimM(G):
    " Calculate the max dimension of matrices M for all edges and (s)"
    max_dim=0;
    for u,v in G.edges():
        A=G[u][v]['As'];
        for s in range(len(A)):
            max_dim=max(A[s].shape) if(max(A[s].shape)>max_dim) else max_dim;       
    return max_dim; 
