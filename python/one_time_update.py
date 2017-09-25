'''

Updates one time steps from the previous one

INPUT: 
	- G : graph containing all the previous 2 time steps A matrices and J interactions
	- P : initial time probability distribution --> needed to build C0
	- svd_routine: 0=fixed number of Singular Values; 1=fixed ratio norms
	- svd_threshold: truncation threshold --> if  svd_routine==0 ==> number of sv to be kept; if svd_routine==1 ==> norm threshold
	- max_n : max number of singular values  --> only used if svd_routine==0
	- dynamic_type: type of dynamic rule --> defines transition matrix W
	- d : spin dimension, i.e. # of possible spin variables (e.g. d=2 ==> s \in {0,1} )
	- threshold : epidemic threshold= # number of neighboring spins infected --> only uesd if dynamic_type==2
	- Q : random bias for majority rule --> only used if dynamic_type==0
	- beta : inverse temperature for Glauber dynamics --> only used if dynamic_type==1
	- T : time step since the beginning of the observed dynamic, is the one for which you want to calculate the update

OUTPUT: 
	- sv_used_ratio : minimum over different A matrices of singular value ratio:= n/(n+sv_rejected)
	- norm_ratio: maximum over different A matrices of the quantity ratio:=[1- trunc_norm/exact_norm]
		where: 
			trunc_norm= \sum_l (s[l]**2);   l=1,...,n   ; n=Number of used singular values
			exact_norm=np.linalg.norm(s);   complete norm, i.e. same as trunc_norm but with n= Total number of singular values
	
'''


import numpy as np
import numpy.linalg as linalg
from copy import deepcopy,copy
import math as math
import random
from collections import deque # Shift element

import tools_dynamics as tl_dyn
infty=10**6



def C_from_A(z,W,A,d):
	'''
	
	Calculates matrices C's from matrices A's
		
	INPUT:
		- z=degree of node u; 
		- w[i]=d-dim array with entries G.node[u]['w'], np array with dim=(*((z+1)*(d,)))  ;
		- A=(z-1)-dim array with entries G[k][u]['As'][t][:,i,:] );
	OUTPUT:
		- C's in an array
	'''
	if(len(A)==0): return np.array(W).reshape(d,d,1,1);
	c=d*[0];
	for i in range(len(W)):   # sigma_i
		#  First contraction to calculate C
		n=1;
		wA=np.reshape(W[i],(d,d,d**(z-2),1,1));
		for k in range(len(A)):    
			iA=min(i,A[k].shape[1]-1) # takes care of the faked index d=1
			wA=np.einsum('jesAB,eab->jsAaBb',wA,A[k][:,iA,:]);
			if n<=z-2 :
				dims=wA.shape
				wA=np.reshape(wA,(d,d,d**(z-2-n),dims[2]*dims[3],dims[4]*dims[5] ));
				n+=1;
		dims=wA.shape;
		wA=np.reshape(wA,(d,dims[2]*dims[3],dims[4]*dims[5]) );               
		c[i]=wA;

	return np.array(c)


def svd_and_truncation(M,routine=0,threshold=0, max_n=infty):
	''' 
	Performs first SVD and then truncate
	INPUT:
		- routine : 0=fixed number of SV; 1=fixed ratio norms 
		- threshold : truncation threshold	 
		- M : matrix to be decomposed 
		- output : $4=# SV used; $5= # sv rejected; $6=ratio

	OUTPUT:
		- U,s,V : Singular Value Decomposition matrices
		- n : # of singular values used
		- sv_rejected : # of singular not used because truncated
		- ratio : # norm ratio 1.-math.sqrt(trunc_norm/exact_norm)

	'''

	U,s,V=linalg.svd(M, full_matrices=False);
	s/=np.linalg.norm(s); 
	exact_norm=np.linalg.norm(s);  # complete norm
	trunc_norm=0.;  # truncated norm

	if(routine==0 and threshold==0):  # NO TRUNCATION
		return U,s,V,s.shape[0],0,0.;
	
	n=0; # Number of used singular values
	sv_rejected=0; # number of not used singular values
	ratio=0. # 1- trunc_norm/exact_norm

	if(routine==0 and threshold>0): # fixed number of singular values
		n=min(int(threshold),s.shape[0]);
		sv_rejected=s.shape[0]-n;
		for l in range(n):trunc_norm+=s[l]*s[l];
		ratio=1.-math.sqrt(trunc_norm/exact_norm);

	if(routine==1):  # Fixed ratio btw norms			
		ratio=1.-threshold; # % of total norm that has to be preserved by the truncation
		while( (trunc_norm<exact_norm*(ratio**2) or n<min(10,s.shape[0])) and (n<min(max_n,s.shape[0])) ): 
			trunc_norm+=s[n]*s[n];
			n+=1;
		sv_rejected=s.shape[0]-n;	
		ratio=1.-math.sqrt(trunc_norm/exact_norm);

	if(n>0):
		s=s[:n];
		U=U[:,:n];
		V=V[:n,:];

	return U,s,V, n, sv_rejected, ratio
#  ----------- ----------- ----------- ----------- -----------

# Calculate C's				
def update(dynamic_type,beta,G,d,T,P,svd_routine=0, svd_threshold=0, max_n=infty,Q=0.,threshold=0):
	''' 
	Updatea all edges
	'''
	sv_used_ratio=1. ; norm_ratio=0.;

	for u in G.nodes():
	
		z=G.out_degree(u);

		for n in range(z):
			v=G.neighbors(u)[n];
			Cs=(T+2)*[0]
			G[u][v]['Bs']=(T+3)*[0]
			
			k_minus_j=set(G.neighbors(u))-set([v]);     # a part from j
			# INPUT variables for function A->C 
			if(dynamic_type==1):
				J=[G[u][v]['J']]; 
				JJ=[ G[u][k]['J'] for k in k_minus_j];   #  -------   k in k_minus_j 	---------
				J=np.append(J,JJ);
				W=[tl_dyn.w_glauber(u,d,G,beta,J)[i,:] for i in range(d) ]; # each W[i] has z indexes
			elif(dynamic_type==0):
				W=[tl_dyn.w_majority(u,d,G,Q)[i,:] for i in range(d) ]; # each W[i] has z indexes	
			elif(dynamic_type==2):
				threshold=z-1;
				W=[tl_dyn.w_epidemic(u,d,G,threshold)[i,:] for i in range(d) ]; # each W[i] has z indexes	
				

			#	----------------------------------------------------------------------------------------------------------------------------------
			# 	IMPORTANT :A's and W's have to be ordered in the SAME way!!!    -->  k in k_minus_j  =>  careful with the J's entering W
			#	----------------------------------------------------------------------------------------------------------------------------------

			# Calculate the C(s) for t=1,...,T+1
			for t in range(1,T+2):
				A=[G[k][u]['As'][t] for k in k_minus_j ];    #  -------   k in k_minus_j 	---------
				Cs[t]=C_from_A(z,W,A,d);
				
			# Calculate C0
			C0=d*[0]
			for i in range(d): 
				C0[i]=np.array(P[u][i]);
				C0[i]=C0[i].reshape(-1,1);
				for k in k_minus_j:
					C0[i]=np.outer(C0[i],G[k][u]['As'][0][i,:]);
					C0[i]=C0[i].reshape(-1,1);
			Cs[0]=np.array(C0)
			
			# ---- Define auxiliary matrices
			D=(T+2)*[0];
			
			#------------------------------------------------------------------------

			# Preparation of Right ON basis: sweep R->L with only SVD
			#   ---------  NO TRUNCATION!    ---------  
			# Right boundary
			X=Cs[0].copy()

			X=np.einsum('iab->aib',X).reshape(X.shape[1],d*X.shape[2])
			U,s,Cs[0]=linalg.svd(X, full_matrices=False);
			E=np.einsum('ak,k->ak',U,s);
			# Bulk
			for t in range(1,T+1):
				X=Cs[t].copy();
				Y=np.einsum('ijab,bk->aijk',X,E);
				Y=Y.reshape(X.shape[2],d*d*E.shape[1]);
				U,s,Cs[t]=linalg.svd(Y, full_matrices=False);
				Cs[t]=np.einsum('aijb->iajb',Cs[t].reshape(Cs[t].shape[0],d,d,-1));
				E=np.einsum('ak,k->ak',U,s);
			
			# Left boundary
			Cs[T+1]=np.einsum('ijab,bk->iajk',Cs[T+1],E);

			#------------------------------------------------------------------------
			n=0;sv_rejected=0; ratio=0.;
			#-----------------------------------------------------------------------
			# 1. Left to Right sweep

			# Left boundary
			X=Cs[T+1].copy();
			size=X.shape;
			X=X.reshape(d*X.shape[1],d*X.shape[3]);

			U,s,V,n,sv_rejected,ratio=svd_and_truncation(X,svd_routine,svd_threshold,max_n)
			
			sv_used_ratio=min((float(n)/float(n+sv_rejected)) , sv_used_ratio);
			norm_ratio=max(norm_ratio,ratio);

			D[T+1]=U.reshape(d,1,size[1],s.shape[0]);
			E=np.einsum('k,kb->kb',s,V);
			E=E.reshape(V.shape[0],d,size[3]);
			# Bulk
			for t in range(T,0,-1):
				size=Cs[t].shape;
				Y=np.einsum('kjb,ibsa->ijksa',E,Cs[t]);
				Y=Y.reshape(d*d*s.shape[0],d*size[3]);
	
				U,s,V,n,sv_rejected,ratio=svd_and_truncation(Y,svd_routine,svd_threshold,max_n)
				sv_used_ratio=min((float(n)/float(n+sv_rejected)) , sv_used_ratio);
				norm_ratio=max(norm_ratio,ratio);

				D[t]=U.reshape(d,d,-1,s.shape[0]);
				E=np.einsum('k,kb->kb',s,V);
				E=E.reshape(V.shape[0],d,size[3]);
			# Right boundary
			X=Cs[0].reshape(Cs[0].shape[0],d,-1);
			D[0]=np.einsum('kja,aib->ikjb',E,X);
			D[0]=D[0].reshape(d*D[0].shape[1],-1);
			Cs=[]


			#-----------------------------------------------------------------------

			#-----------------------------------------------------------------------
			# Second sweep R ->L ---
			# Right boundary

			U,s,V,n,sv_rejected,ratio=svd_and_truncation(D[0],svd_routine,svd_threshold,max_n)
			sv_used_ratio=min((float(n)/float(n+sv_rejected)) , sv_used_ratio);
			norm_ratio=max(norm_ratio,ratio);

			V=V.reshape(V.shape[0],d,-1);
			G[u][v]['Bs'][0]=np.einsum('kjb->jkb',V).reshape(d,-1,1);
			E=np.einsum('ak,k->ak',U,s);
			E=E.reshape(d,-1,s.shape[0]);
			# Bulk
			for t in range(1,T+1):
				Y=np.einsum('sjka,iab->skijb',D[t],E);
				Y=Y.reshape(d*Y.shape[1],d*d*s.shape[0]);
				
				U,s,V,n,sv_rejected,ratio=svd_and_truncation(Y,svd_routine,svd_threshold,max_n)
				sv_used_ratio=min((float(n)/float(n+sv_rejected)) , sv_used_ratio);
				norm_ratio=max(norm_ratio,ratio);

				V=V.reshape(s.shape[0],d,d,-1);
				G[u][v]['Bs'][t]=np.einsum('kijb->ijkb',V);
				E=np.einsum('ak,k->ak',U,s);
				E=E.reshape(d,-1,s.shape[0]);
			
			# Boundaries T, T+1
			Y=np.einsum('sab->asb',E).reshape(E.shape[1],-1);
	
			U,s,V,n,sv_rejected,ratio=svd_and_truncation(Y,svd_routine,svd_threshold,max_n)
			sv_used_ratio=min((float(n)/float(n+sv_rejected)) , sv_used_ratio);
			norm_ratio=max(norm_ratio,ratio);

			V=V.reshape(s.shape[0],d,1,-1);
			G[u][v]['Bs'][T+1]=np.einsum('kijb->ijkb',V);
			E=np.einsum('ak,k->ak',U,s);
			G[u][v]['Bs'][T+2]=np.einsum('sjak,kb->sjab',D[T+1],E);	

	for u,v in G.edges():
		G[u][v]['OldA']=deepcopy(G[u][v]['As']);
		G[u][v]['As']=deepcopy(G[u][v]['Bs'])
		del G[u][v]['Bs']

	return sv_used_ratio,norm_ratio;
