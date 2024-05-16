from copy import deepcopy
import networkx as nx
import matplotlib
from matplotlib import pyplot as plt
import scipy as sp
import sys
import tkinter as tk
from tkinter import messagebox
import json


graph ={0:[1,2],
		1:[0,3],
		2:[0,3],
		3:[1,2]}

		
graphtest={0:[3],
	   1:[2,3],
	   2:[1],
	   3:[0,1],
	   4:[5,6],
	   5:[4,6],
	   6:[4,5]}

bipartite_graph = {
    (0,1): [(0,4), (0,5)],
    (0,2): [(0,4), (0,6)],
    (0,3): [(0,5),(0,6)],
    (0,4): [(0,1),(0,2)],
    (0,5): [(0,1),(0,3)],
    (0,6): [(0,2),(0,3)]
}







def read_graph_from_file(file_path):
    with open(file_path, 'r') as file:
        graph_data = json.load(file)
        print("GRAPH DATA IN 44",graph_data)
    return graph_data

def make_g(graph_data):
	g={}
	
	for edge in graph_data["edges"]:
		g[edge["source"]]=edge["target"]
		
	print("here is g", g)
	return(g,graph_data["neighbours"])


def makeGx(graphx,x,neighbours_x):
	graphx[x] = neighbours_x 
	for r in neighbours_x:
		if r in graphx:
			graphx[r].append(x)
		else:
			graphx[r] = [x]

	
	return graphx

def makeX(graphx,x,neighbours_x):
	new_graph_x = {}
	new_graph_x[x] = neighbours_x

	for i in neighbours_x:
		if i in new_graph_x:
			new_graph_x[i].append(x)
			for y in graphx[i]:
				if y in neighbours_x:
					new_graph_x[i].append(y)
		else:
			new_graph_x[i]=[x]
			for y in graphx[i]:
				if y in neighbours_x:
					new_graph_x[i].append(y)
	print("new graph xn",new_graph_x)
	
	return(new_graph_x)
	



def is_bipartite(grap):
	# Initialize colors dic with -1 (unvisited)
	global sets 
	sets = {0: set(), 1: set()}
	colors={}
	#sets = {0: set(), 1: set()}
	
	for key in grap:
		
		colors[key]=-1
	
	# start DFS from vertex 0 with color 0
	#start= list(grap.keys())[0]
	
	for start in grap:
		if colors[start] == -1:
			stack = [(start, 0)]
			#initializes a stack with a tuple containing the start of dfs and its color.
			while stack:
			
				vertex, color = stack.pop()
				if colors[vertex]==-1:
					colors[vertex]= color
			
					sets[color].add(vertex)
				

				for vals in grap.get(vertex): #graph[vertex] gives neighbors
					
					if colors.get(vals) == color:
						
						# Two adjacent vertices have the same color, not bipartite
						#print("False")
						return False
					elif colors.get(vals) == -1:
							# Neighbor is unvisited, assign opposite color and add to stack
						stack.append((vals, 1-color))
							
			
	
	#print("graph is bipartite")
	#print("SETS",sets)
	return True




def incomp(graph):
	
	
	incomp = {}
	

	x= range(len(graph))

	for i in graph:
		for neighbour in graph.get(i):

			for neigh in graph[neighbour]:
				
				if neigh==i:
					
					continue

				if ((i,neighbour)) in incomp :
				
					if (neighbour,neigh) not in incomp[(i,neighbour)] and i not in graph[neigh]:
						incomp[(i,neighbour)].append((neighbour,neigh))
							
					if (neighbour,i) not in incomp[(i,neighbour)]:
						incomp[(i,neighbour)].append((neighbour,i))
				else:
					if i not in graph[neigh]:	
						incomp[(i,neighbour)]=[(neighbour,neigh)]
						incomp[(i,neighbour)].append((neighbour,i))
					else:
						incomp[(i,neighbour)]=[(neighbour,i)]
						


				if ((neighbour,neigh)) in incomp :
					if (neigh,neighbour) not in incomp[(neighbour,neigh)] :
						incomp[(neighbour,neigh)].append((neigh,neighbour))
					if (i,neighbour) not in incomp[(neighbour,neigh)] and i not in graph[neigh]:
						incomp[(neighbour,neigh)].append((i,neighbour))
					
				else:
						
					
					if i not in graph[neigh]:
						incomp[(neighbour,neigh)]=[(i,neighbour)]
						incomp[(neighbour,neigh)].append((neigh,neighbour))
					else:
						incomp[(neighbour,neigh)]= [(neigh,neighbour)]

				
						
					

	
	return incomp



def make_union_graph(grapha, graphb):
	
	graph_union = {
    key: list(set(grapha.get(key, []) + graphb.get(key, [])))
    for key in grapha.keys() | graphb.keys()
    }
	#print(graph_union)
	
	return graph_union



#def make_union(graph,graphX): #for list of tuples
#	
#	graph_union = graph+graphX
#	graph_union =list(set(graph_union))
	
#	return graph_union



def compute_non_edges(graphx, neighbours_x):
	non_edges = []
	x = len(graphx)-1
	edges_x =[]
	y= range(len(graphx))

	for i in neighbours_x:
		for j in graphx.get(i):
			if j not in neighbours_x and j!=x:
				non_edges.append((x,j))
				non_edges.append((j,x))
	
	return non_edges

def mmc(H,graphB,nonEdges):
	print("this is H", H)
	print("this is graphB:")
	print(graphB)

	h_current=deepcopy(H)
	
	currentB=deepcopy(graphB)
	marked=[]
	nonedges=deepcopy(nonEdges)

	

	
	
	while len(nonedges) >0:

		current_non_edge=nonedges.pop()
		nonedges.append(current_non_edge)

		print("CURRENT NON EDGE", current_non_edge)
		
		x=current_non_edge[0]
		v=current_non_edge[1]
		
		NHx=h_current[x]
		NHv=h_current[v]
		
		
		itsc_x_v= list(set(NHx)&set(NHv)) #intersection
		#print("INYTER", itsc_x_v)
		
		Chxu={}
		Chxu1={}
		Chxu2={}
		
		for w in itsc_x_v:
			if w!=x and w!=v:

			
				if (x,w) in Chxu1 and (w,v) not in Chxu1[(x,w)]:
					Chxu1[(x,w)].append((w,v))
				if (w,v) in Chxu1 and (x,w) not in Chxu1[(w,v)]:
					Chxu1[(w,v)].append((x,w))
				if (x,w) not in Chxu1:
					Chxu1[(x,w)]=[(w,v)]
				if (w,v) not in Chxu1:
					Chxu1[(w,v)]=[(x,w)]

				if (v,w) in Chxu2 and (w,x) not in Chxu2[(v,w)]:
					Chxu2[(v,w)].append((w,x))
				if (w,x) in Chxu2 and (v,w) not in Chxu2[(w,x)]:
					Chxu2[(w,x)].append((v,w))
				if (v,w) not in Chxu2:
					Chxu2[(v,w)]=[(w,x)]
				if (w,x) not in Chxu2:
					Chxu2[(w,x)]=[(v,w)]
		Chxu=make_union_graph(Chxu1,Chxu2)
		print("CHXU286",Chxu)
		


		
		
		B_union_chxu = make_union_graph(currentB,Chxu)
		print("b union chxu", B_union_chxu)


			
		if is_bipartite(B_union_chxu):
			print("Bunion ch is bipartite")
			
			currentB = B_union_chxu
			


		else:
			print("Bunion ch is not b")
			if x!=v:
				if x in h_current and v not in h_current[x]:
					h_current[x].append(v)
				if v in h_current and x not in h_current[v]:
					h_current[v].append(x)
				if x not in h_current:
					h_current[x]=v
				if v not in h_current:
					h_current[v]=x
			
				if (x,v) in currentB and (v,x) not in currentB[(x,v)]:
					currentB[(x,v)].append((v,x))
				elif (x,v) not in currentB:
					currentB[(x,v)] = [(v,x)]
				if (v,x) in currentB and (x,v) not in currentB[(v,x)]:
					currentB[(v,x)].append((x,v))
				elif(v,x) not in currentB:
					currentB[(v,x)]=[(x,v)]
				
			Nhx = h_current[x]
			

			Nhv = h_current[v]
			print("Nhx", Nhx)
			print("Nhv", Nhv)
			#for i in currentB:
			#	print(type(currentB[i]))
			
			for z in Nhx: #step 11
				if z not in Nhv and z!=x and z!=v:
					print("ddfsdf",(v,x),(x,z),(z,x),(x,v))
					if (v,x) in currentB and (x,z) not in currentB[(v,x)]:
						currentB[(v,x)].append((x,z))
					if (x,z) in currentB and (v,x) not in currentB[(x,z)]:
						currentB[(x,z)].append((v,x))
					if (z,x) in currentB and (x,v) not in currentB[(z,x)]:
						currentB[(z,x)].append((x,v))
					if (x,v) in currentB and (z,x) not in currentB[(x,v)]:
						currentB[(x,v)].append((z,x))
					if (v,x) not in currentB:
						currentB[(v,x)]=[(x,z)]
					if (x,z) not in currentB:
						currentB[(x,z)]=[(v,x)]
					if (z,x) not in currentB:
						currentB[(z,x)]=[(x,v)]
					if (x,v) not in currentB:
						currentB[(x,v)]=[(z,x)]
			for u in Nhv:
				if u not in Nhx and u!=v and u!=x:
					if (x,u) in marked:
						print("x,u in marked",(x,v),(v,u), (u,v),(v,x))
						if (x,v) in currentB and (v,u) not in currentB[(x,v)]:
							currentB[(x,v)].append((v,u))
						if (v,u) in currentB and (x,v) not in currentB[(v,u)]:
							currentB[(v,u)].append((x,v))
						if (u,v) in currentB and (v,x) not in currentB[(u,v)]:
							currentB[(u,v)].append((v,x))
						if (v,x) in currentB and (u,v) not in currentB[(v,x)]:
							currentB[(v,x)].append((u,v))
						if (x,v) not in currentB:
							currentB[(x,v)]=[(v,u)]
						if (v,u) not in currentB:
							currentB[(v,u)]=[(x,v)]
						if (u,v) not in currentB:
							currentB[(u,v)]=[(v,x)]
						if (v,x) not in currentB:
							currentB[(v,x)]=[(u,v)]

					elif (x,u) not in nonedges and x!=u and (x,u)!=(x,v):
						print("not in non edges, append in non edges", (x,u) )
						nonedges.append((x,u))
		print("currentB end of iteration", currentB)
		print("current h end of iteration", h_current)
		marked.append(current_non_edge)
		
		nonedges.remove(current_non_edge)
		

		
	

	return (h_current, currentB)







def main():
	user_input_neighbours_x=[]
	file_path = 'graph_data.json'
	graph_data = read_graph_from_file(file_path)
	g,user_input_neighbours_x= make_g(graph_data)
	x= len(g) #new vertex
	
	
	graphX = deepcopy(g)
	
	#user_input_neighbours_x = [1,2,3] #fixed input



	gx= makeGx(graphX,x,user_input_neighbours_x)

	new_x = makeX(graphX, x, user_input_neighbours_x)
	#print("NEW", new_x)
	


	#if is_bipartite(g):
	incompatability_graph_g = incomp(g)

		#if is_bipartite(incompatability_graph_g):
			#print("INC OF G IS bipartite")
		
		#incompatibility_graph_X= incomp(gx)
		
	incompatibility_graph_X= incomp(new_x)
	#print("inc x", incompatibility_graph_X)
		
		#print("INCOPx",incompatibility_graph_X)

		#if is_bipartite(incompatibility_graph_X):
			#print("INC OF Gx IS bipartite")
		
		

	graphB = make_union_graph(incompatability_graph_g,incompatibility_graph_X)
	#print("partial i", graphB)	
	L_non_edges = compute_non_edges(gx,user_input_neighbours_x) #list of non edges
	print("L_non_edges",L_non_edges)


	H=deepcopy(gx)
		
	Hh, Bh=mmc(H,graphB,L_non_edges)

		
		
	is_bipartite(Bh) #set color sets
	print("BH", Bh)
	G=nx.Graph()
	g=nx.Graph()
	B_first_partition_nodes=[]
	for (i,x) in Bh:
				
		g.add_node((i,x))
		for (y,j) in Bh[(i,x)]:
			g.add_edge((i,x),(y,j))
				
		
	print("H:", Hh)
	for i in Hh:
		G.add_node(i)
		for y in Hh[i]:
			if i not in graph:
				G.add_edge(i,y,color='r')
			else:
				G.add_edge(i,y,color='b')

	colors = nx.get_edge_attributes(G,'color').values()
		
	plt.figure(1)
		

	nx.draw_networkx(g,pos = nx.drawing.layout.bipartite_layout(g, sets[1]),node_size=800)
		
	plt.figure(2)
	nx.draw_planar(G, with_labels = True, edge_color=colors)
		
		
	plt.show()
	
	

if __name__ == "__main__":
    main()









	
