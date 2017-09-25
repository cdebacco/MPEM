'''
Functions needed to manipulate the main.py
'''

import networkx as nx
#----------------------------------------------------------------------
def usage():
	" Help legenda for input parameters"
	print '-h or --help gives you the parameters explanation'
	print '-g sets the graph_type: 0=Reg, 1=ER, 2=WS, 3=Power-Law tree, 4=Reg_ER; 5=balanced_tree(k,n)'
	print '-d sets graph connected component: 0=Keep the whole graph; 1=Keep only the max connected component'
	print '-n sets graph size'
	print '-k sets average degree'
	print '-t sets T_max'
	print '-s sets svd_routine: 0=fix the number of singular values; 1=fix the norm cut of ratio. '
	print '-p sets the svd_parameter: either the max number of singular values or the norm cutoff ratio'
	print '-i sets random number generator seed'
	print '-m sets max number of singular values accepted'
	print '-J sets mean value Js'
	print '-j sets variance Js'
	print '-b sets bias spin'
	print '-B sets beta'
	print '-e sets type of dynamic 0=majority; 1=Glauber'
	return 1;

def output_parameters(outfile,G,d_component,n,k,T_max,svd_routine,svd_parameter,seed,max_n,bias,dynamic_type,beta,J0,J2):
	print >> outfile, '-d', d_component;
	print >> outfile, '-n', n;
	print >> outfile, '-k', k;
	print >> outfile, '-t', T_max;
	print >> outfile, '-s', svd_routine;
	print >> outfile, '-p', svd_parameter;
	print >> outfile, '-i', seed;
	print >> outfile, '-m', max_n;
	print >> outfile, '-b', bias;
	print >> outfile, '-e', dynamic_type;
	if(dynamic_type==1):
		print >> outfile, '-B', beta;
		print >> outfile, '-J', J0;
		print >> outfile, '-j', J2;
	print >> outfile, 'G.info=', nx.info(G);