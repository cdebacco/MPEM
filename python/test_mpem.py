#!/usr/bin/python


'''

Simulates the dynamics by using the MPS equations of motion

INPUT : see usage()

OUTPUT : prints on file the evolution over time of magnetization, energy, overlap and correlation

'''

import numpy as np
import networkx as nx
import random
import one_time_update as otu 		 # contains MPS update routines (SVD and truncation)
import math as math
import time
import sys, getopt  # Command line argument

import tools_main as tl_main
import tools_observables as tl_obs
import tools_dynamics as tl_dyn

# --------------- Global parameters -----------------------------------
iseed=0;   # To assign J's
random.seed(iseed);
infty=10**6;

d=2;  # dimension of spin space
Mmin=3; # dimension of matrices A
Mmax=6; # dimension of matrices A
T_max=10; # max iteration time
T=0;
svd_routine=1 #  0=fixed number of SV; 1=Fixed ratio truncated_norm/exact_norm
svd_parameter=3 # if svd_routine==0 -> number of SV; if svd_routine==1 -> 1-truncatec
max_n=250;   # max number of accepted singular values

# --------------- Random graph ---------------
d_component=1; # 0=Allow disconnected components; 1=Keep only giant component
g_name='reg'    # used in naming output files
n=1000			# number of nodes
k=3				# average degree

# ------- Dynamics parameters  ----------------------------------------
J0=0.; # mean value of J's
J2=1.0; #1./float(n); # variance of J's
bias=0.05 # Initial bias over the plus (+1) sigma's
beta=1.0;	# Used in calculating energy and glauber dynamics transition matrix
gamma=2.3	# Used only if using power law graphs

seed=0;   # To build the graph
dynamic_type=1;  # 0=Majority; 1=Glauber

def main(argv):
	" main function"	
	global g_name,d_component,n,k,T_max,svd_routine,svd_parameter,seed,max_n,J0,J2,bias,beta,dynamic_type;
	H=nx.Graph();
	try:
		opts, args=getopt.getopt(argv,"hg:d:n:k:t:s:p:i:m:J:j:b:B:e:","help");
	except getopt.GetoptError:
		tl_main.usage();
		sys.exit(2);	
	for opt,arg in opts:
	 	if opt in("-h","--help"):
	 		tl_main.usage();
	 		sys.exit();
	 	elif opt=='-d': d_component=int(arg);
	 	elif opt=='-n': n=int(arg);
	 	elif opt=='-k': k=int(arg);
	 	elif opt=='-t': T_max=int(arg);
	 	elif opt=='-s': svd_routine=int(arg);
	 	elif opt=='-p': svd_parameter=int(arg);
	 	elif opt=='-i': seed=int(arg);
		elif opt=='-m': max_n=int(arg);
		elif opt=='-J': J0=float(arg);
		elif opt=='-j': J2=float(arg)/float(n);
		elif opt=='-b': bias=float(arg);
		elif opt=='-B': beta=float(arg);
		elif opt=='-e': dynamic_type=int(arg);
	
	p=float(k)/float(n);
	if(svd_routine==1): svd_parameter=float(10.**(-svd_parameter));
	print'svd_parameter',svd_parameter
	# GENERATE GRAPH 			
	H=nx.random_regular_graph(k, n, seed); g_name='reg';
	

	out_dir='../data/';
	
	if(dynamic_type==1):
		outfile = open(out_dir+g_name+str(n)+'k'+str(k)+'s'+str(svd_routine)+str(svd_parameter)+'e'+str(dynamic_type)+'b'+str(bias)+'B'+str(beta)+'dyn.dat', 'w')
		out_svd=open(out_dir+g_name+str(n)+'k'+str(k)+'s'+str(svd_routine)+str(svd_parameter)+'e'+str(dynamic_type)+'b'+str(bias)+'B'+str(beta)+'svd.dat', 'w')
		histo_file=open(out_dir+g_name+str(n)+'k'+str(k)+'s'+str(svd_routine)+str(svd_parameter)+'e'+str(dynamic_type)+'b'+str(bias)+'B'+str(beta)+'histo.dat', 'w')
		out_parameters=open(out_dir+'parameters_'+g_name+str(n)+'k'+str(k)+'s'+str(svd_routine)+str(svd_parameter)+'e'+str(dynamic_type)+'b'+str(bias)+'B'+str(beta)+'parameters.dat', 'w')
		
	elif(dynamic_type==0):
		outfile = open(out_dir+g_name+str(n)+'k'+str(k)+'s'+str(svd_routine)+str(svd_parameter)+'e'+str(dynamic_type)+'b'+str(bias)+'dyn.dat', 'w')
		out_parameters=open(out_dir+'parameters_'+g_name+str(n)+'k'+str(k)+'s'+str(svd_routine)+str(svd_parameter)+'e'+str(dynamic_type)+'b'+str(bias)+'parameters.dat', 'w')
		out_svd=open(out_dir+g_name+str(n)+'k'+str(k)+'s'+str(svd_routine)+str(svd_parameter)+'e'+str(dynamic_type)+'b'+str(bias)+'svd.dat', 'w')
	
	
	print H,n,k, g_name
	if(d_component==1):
		print 'Connected components',nx.number_connected_components(H);
		H=sorted(list(nx.connected_component_subgraphs(H)), key = len, reverse=True)[0];  # Giant component is the first element of the list
		H=nx.convert_node_labels_to_integers(H);

	G=nx.DiGraph(H) 
	if(dynamic_type==1):sumJ=tl_dyn.assign_J(J0,J2,G,histo_file);  # Fix disorder for Glauber dynamics

	tl_main.output_parameters(out_parameters,G,d_component,n,k,T_max,svd_routine,svd_parameter,seed,max_n,bias,dynamic_type,beta,J0,J2); out_parameters.close();

	degree_sequence=sorted(nx.degree(G).values(),reverse=True) # degree sequence
	dmax=max(degree_sequence)/2 if(nx.is_directed(G)) else max(degree_sequence);
	print 'k_max=', dmax
	print 'J2,J0 ' , J2,J0

	# Initial Probability distribution  -------
	P=np.array([ [0.5-bias,0.5+bias]  for u in G.nodes() ]); 

	# INITIALIZE observables ---------------------------
	m=[0. for t in range(T_max)]; # magnetization
	q=[0. for t in range(T_max)]; # EA parameter (overlap)
	Cor=[0. for t in range(T_max)]; # spin-spin correlation
	Z=G.number_of_nodes();

	if(dynamic_type==0):
		for u in G.nodes():
			G.node[u]['w']=tl_dyn.w_majority(u,d,G);
			m[0]+=(P[u][1]-P[u][0]);Z+=1;
			q[0]+=(P[u][1]-P[u][0])*(P[u][1]-P[u][0]);
		m[0]/=float(Z);q[0]/=float(Z);
	elif(dynamic_type==1):
		for u in G.nodes():
			J=[ G[u][v]['J'] for v in G.neighbors(u) ];
			G.node[u]['w']=tl_dyn.w_glauber(u,d,G,beta,J);
			m[0]+=(P[u][1]-P[u][0]);Z+=1;
			q[0]+=(P[u][1]-P[u][0])*(P[u][1]-P[u][0]);
		m[0]/=float(Z);q[0]/=float(Z);Z=0.;	
	
	Cor[0]=0.;  #     <==  We start from a factorized initial condition		
	
	# INITIALIZE A's, M's, C's
	for u,v in G.edges():
		M=[random.randrange(Mmin,Mmax) for t in range(T+2)]          
	# messages i --> j
		M[0]=1;M[T]=1;
	# If T=0 just need to initialize A(0) and A(1)
		G[u][v]['As']=[np.random.rand(d,M[0],1)];
		G[u][v]['As'].append(np.random.rand(d,1,1,M[T]));

		G[u][v]['OldA']=[np.random.rand(d,M[0],1)];

		G[u][v]['As'][0][0,:,:]=1.#P[v][0];
		G[u][v]['As'][0][1,:,:]=1.#bP[v][1];
		G[u][v]['As'][1][0,:,:,:]=P[u][0];
		G[u][v]['As'][1][1,:,:,:]=P[u][1];

		G[u][v]['OldA']=G[u][v]['As'][0];

	for u in G.nodes():	G.node[u]['marginal']=P[u];

	sv_ratio=1.;norm_ratio=0.;
	t0 = time.time();t1=t0;
	for t in range(T,T_max):
			
		print >> out_svd, t, tl_obs.max_dimM(G),sv_ratio,norm_ratio,str(t1-t0);out_svd.flush();

		if(t>0):	
			out_mag = open(out_dir+g_name+str(n)+'k'+str(k)+'s'+str(svd_routine)+str(svd_parameter)+'t'+str(t)+'b'+str(bias)+'B'+str(beta)+'mag.dat', 'w')
			out_cor = open(out_dir+g_name+str(n)+'k'+str(k)+'s'+str(svd_routine)+str(svd_parameter)+'t'+str(t)+'b'+str(bias)+'B'+str(beta)+'corr.dat', 'w')
			Cor[t]=tl_obs.calulate_marginals(G,d,out_mag,out_cor);	
			out_mag.close();out_cor.close();

		t1 = time.time()	
		m[t],q[t]=tl_obs.calulate_observable(G,d);

		sv_ratio,norm_ratio=otu.update(dynamic_type,beta,G,d,t,P,svd_routine=svd_routine,svd_threshold=svd_parameter,max_n=max_n);
		
		t2 = time.time()
		time_diff = round(t2 - t1)
		
		print >> outfile, t,m[t], q[t], Cor[t], str(t1-t0); outfile.flush();
		print 't = ', t, ' calculated in ',time_diff, 's';
		print ' <m>= ',m[t], ' q=', q[t], 'Cor=', Cor[t], 
		print 'maxMdim=',tl_obs.max_dimM(G),' sv_ratio=',sv_ratio,' norm_ratio=',norm_ratio;

	#  END CYCLE OVER t	
	time_diff = round(time.time() - t0)
	minute = time_diff / 60
	seconds = time_diff % 60  # Same as time_diff - (minutes * 60)
	print 'Total time=', minute, 'm and', seconds, 's'

	outfile.close();
	out_svd.close();
#  END MAIN --------------------------------------------------------------
if __name__ == "__main__":
   main(sys.argv[1:])
