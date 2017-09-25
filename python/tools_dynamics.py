'''

Defines functions needed for specifing the dynamical rules

	- w(s|s')  Transition matrix
	- For Glauber dynamics : assign_J 

'''

import numpy as np
import math
import networkx as nx

def w_majority(u,d,G):
	"transition probability: majority rule"
	if(d!=2):
		print ' d is not 2! Reset d accordingly!'
		return False;
	else:	
		z=G.out_degree(u)
		x=np.zeros((z+1)*[d,])
		# Evaluate x[0][d] i.e. w(s_i| neighbors)
		for s in range(d):
			y=x[s,:].copy()
			for i in np.ndindex(y.shape):
				sum=0
				for c in range(len(i)):
					sigma=-1 if(i[c]==0) else 1;
					sum=sum+sigma;
				if(sum>0 and s==1):
					y[i]=1;
				if(sum>0 and s==0):	
					y[i]=0;	
				if(sum<0 and s==1):
					y[i]=0;
				if(sum<0 and s==0):
					y[i]=1;	 
				if(sum==0):
					y[i]=0.5;
			x[s,:]=y
		return x;


def w_glauber(u,d,G,beta,J):
	"transition probability: Glauber dynamic"
	if(d!=2):
		print ' d is not 2! Reset d accordingly!'
		return False;
	else:	
		z=G.out_degree(u)
		x=np.zeros((z+1)*[d,])
		# Evaluate x[0][d] i.e. w(s_i| neighbors)
		for s in range(d):
			y=x[s,:].copy()
			spin=1 if(s==1) else -1;
			for i in np.ndindex(y.shape):
				h=0
				for c in range(len(i)):
					sigma=-1 if(i[c]==0) else 1;
					h=h+sigma*J[c];
					#h=h+sigma*J;
				y[i]=math.exp(beta*spin*h)/(math.exp(-beta*h)+math.exp(beta*h))
			x[s,:]=y
		return x;				


def assign_J(J0,J2,G,histo_file,iseed=0):
	'Input mean j0 and variance J2 of a gaussian distibution'
	r=np.random.RandomState(iseed);
	J_array=np.array([J0]);
	for u,v in G.edges():
		G[u][v]['J']=r.normal(J0,math.sqrt(J2));
		if(nx.is_directed(G)):G[v][u]['J']=G[u][v]['J'];
	sumJ=0.;
	for u,v in G.edges():	
		J_array= np.append(J_array,G[u][v]['J']);
		sumJ+=G[u][v]['J'];
		if(G[u][v]['J']!=G[v][u]['J']): print "Assymmetric coupling!"

	frequency,histo_range=np.histogram(J_array);	
	for i in range(len(histo_range)-1):
		print >> histo_file, histo_range[i],frequency[i];
	return sumJ/float(G.number_of_edges());	
	
