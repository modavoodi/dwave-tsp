
# coding: utf-8

# # 7-City Traveling Salesman Problem
# 
# Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin city?
# 
# ## Looking at our Data
# 
# To get our bearings, here is a map of the cities that we are working with.  First we load in information about our data set (latitude and longitude coordinates) and visualize our data set that we're working with.

# In[ ]:


# Load in complete US map with state boundaries
from bokeh.sampledata import us_states
us_states = us_states.data.copy()
del us_states["HI"]
del us_states["AK"]
state_xs = [us_states[code]["lons"] for code in us_states]
state_ys = [us_states[code]["lats"] for code in us_states]

# initialize figure
from bokeh.plotting import *
from bokeh.io import output_notebook
output_notebook()
p = figure(title="Find shortest route that visits each city", 
           toolbar_location="left", plot_width=550, plot_height=350)

# Draw state lines
p.patches(state_xs, state_ys, fill_alpha=0.0,
    line_color='blue', line_width=1.5)

# Load the lat/longs of our cities of interest
Cities=["Albuquerque","Boston","Charlotte","Detroit","Evanston","Frankfort","Gulfport"]
Lat= [-106.650422, -71.058880,-80.843127,-83.045754,-87.687697,-84.873284,-89.092816]
Lon= [35.084386,42.360082,35.227087,42.331427,42.045072,38.200905,30.367420]
map = {"a":"Albuquerque","b":"Boston","c":"Charlotte","d":"Detroit","e":"Evanston","f":"Frankfort","g":"Gulfport"}
from bokeh.models import Circle
p.circle(Lat, Lon, size=10, color='red', alpha=1)

# Display the image
show(p)


# ## Setting up our Graph
# 
# First, we set up our problem as a graph.  Our nodes are the cities our salesman needs to visit, and our edges are the roads between any pair of cities.

# In[ ]:


# Python library for working with graphs
import networkx as nx    

# Begin with an empty graph
G = nx.Graph()           

# Add each city as a node
coord={}
for i in range(7):
    coord[i]=(Lat[i],Lon[i])
    G.add_node(Cities[i],pos=coord[i],label=Cities[i])
    
# Add the distance between each pair of cities as a weighted edge
G.add_weighted_edges_from([("Albuquerque","Boston",2230),("Albuquerque","Charlotte",1631),("Albuquerque","Detroit",1566),("Albuquerque","Evanston",1346),("Albuquerque","Frankfort",1352),("Albuquerque","Gulfport",1204),("Boston","Charlotte",845),("Boston","Detroit",707),("Boston","Evanston",1001),("Boston","Frankfort",947),("Boston","Gulfport",1484),("Charlotte","Detroit",627),("Charlotte","Evanston",773),("Charlotte","Frankfort",424),("Charlotte","Gulfport",644),("Detroit","Evanston",302),("Detroit","Frankfort",341),("Detroit","Gulfport",1027),("Evanston","Frankfort",368),("Evanston","Gulfport",916),("Frankfort","Gulfport",702)])

# Display the stored graph data as an adjacency matrix
print("Graph adjacency matrix:\n")
print(nx.to_numpy_matrix(G))


# ## Setting our Tunable Parameters
# 
# To start, there are a few parameters that we want to set.
# 
# $\textbf{gamma}$ is the Lagrange parameter.  This controls which is more important:  satisfying our constraints or finding the shortest distance.  For this problem, it is crucial that we visit every city on our route, so we should set this parameter to be larger than the greatest distance between two cities.
# 
# $\textbf{chainstrength}$ tells the embedding function how strongly to tie together chains of physical qubits to make one logical qubit.  This should be larger than any other values in your QUBO.
# 
# $\textbf{numruns}$ tells the system how many times to run our problem.  Due to the probabilistic nature of the D-Wave QPU, we should run the problem many times and look at the best solutions found.

# In[ ]:


gamma = 1400
chainstrength = 4700
numruns = 100 
print("Tunable parameters: \n\tGamma: \t\t\t",gamma,"\n\tChain Strength: \t",chainstrength,"\n\tNumber of runs: \t",numruns)


# ## Building our QUBO
# 
# For this problem, we will assign binary variables to the direct paths (or $\textbf{legs}$) between each pair of cities, or in terms of our graph the $\textit{edges}$ in our graph.  These variables denote whether or not we use that direct path between the pair of cities or not.
# 
# More formally, our binary variables are: $$ab =  \left\{ \begin{array}{l} 1 \text{ if the trip includes segment } A \rightarrow B, \\ 0 \text{ if the trip does not include } A \rightarrow B. \end{array} \right.$$
# 
# Since we will store our QUBO in a matrix (linear terms on diagonal, quadratic terms elsewhere), we will place these binary variables as row and column labels in our matrix, as illustrated in the next code cell.  Remember that with Ocean we can label our binary variables with numbers, letters, or even words!

# In[ ]:


print("\t ab \t ac \t ad \t ae \t af \t ag \t bc \t ... \t fg")
print("ab [\t \t \t \t \t \t \t \t \t \t]")
print("ac [\t \t \t \t \t \t \t \t \t \t]")
print("ad [\t \t \t \t \t \t \t \t \t \t]")
print("ae [\t \t \t \t \t \t \t \t \t \t]")
print("af [\t \t \t \t \t \t \t \t \t \t]")
print("ag [\t \t \t \t \t Q\t \t \t \t \t]")
print("bc [\t \t \t \t \t \t \t \t \t \t]")
print(".  [\t \t \t \t \t \t \t \t \t \t]")
print(".  [\t \t \t \t \t \t \t \t \t \t]")
print(".  [\t \t \t \t \t \t \t \t \t \t]")
print("fg [\t \t \t \t \t \t \t \t \t \t]")


# ### Objective Function
# 
# Our goal or objective in this problem is to minimize the total distance travelled.
# 
# Let's denote the distance between city $x$ and city $y$ by $D_{xy}$.  Remember that we stored this information in our graph as the weights on each edge. Using our binary variables, the total distance travelled is then: $$\text{mileage} = D_{ab} \times ab + D_{ac} \times ac + D_{ad} \times ad + \cdots + D_{fg} \times fg.$$  
# 
# This is what we call our $\textbf{objective function}$:  the function that we want to minimize or maximize.
# 
# To put the terms from our objective function into our Q matrix, we place the distance (or edge weight in our graph) into the correct location in the Q matrix.  Since these are all linear terms, they will all be placed on the diagonal of the matrix.

# In[ ]:


# Initialize our Q matrix
from itertools import combinations

Q = {}

for pair1 in list(combinations(['a','b','c','d','e','f','g'], 2)):
    for pair2 in list(combinations(['a','b','c','d','e','f','g'], 2)):
        Q[(pair1[0]+pair1[1],pair2[0]+pair2[1])] = 0

Q[("ab", "ab")] = G["Albuquerque"]["Boston"]['weight']
Q[("ac", "ac")] = G["Albuquerque"]["Charlotte"]['weight']
Q[("ad", "ad")] = G["Albuquerque"]["Detroit"]['weight']
Q[("ae", "ae")] = G["Albuquerque"]["Evanston"]['weight']
Q[("af", "af")] = G["Albuquerque"]["Frankfort"]['weight']
Q[("ag", "ag")] = G["Albuquerque"]["Gulfport"]['weight']
Q[("bc", "bc")] = G["Boston"]["Charlotte"]['weight']
Q[("bd", "bd")] = G["Boston"]["Detroit"]['weight']
Q[("be", "be")] = G["Boston"]["Evanston"]['weight']
Q[("bf", "bf")] = G["Boston"]["Frankfort"]['weight']
Q[("bg", "bg")] = G["Boston"]["Gulfport"]['weight']
Q[("cd", "cd")] = G["Charlotte"]["Detroit"]['weight']
Q[("ce", "ce")] = G["Charlotte"]["Evanston"]['weight']
Q[("cf", "cf")] = G["Charlotte"]["Frankfort"]['weight']
Q[("cg", "cg")] = G["Charlotte"]["Gulfport"]['weight']
Q[("de", "de")] = G["Detroit"]["Evanston"]['weight']
Q[("df", "df")] = G["Detroit"]["Frankfort"]['weight']
Q[("dg", "dg")] = G["Detroit"]["Gulfport"]['weight']
Q[("ef", "ef")] = G["Evanston"]["Frankfort"]['weight']
Q[("eg", "eg")] = G["Evanston"]["Gulfport"]['weight']
Q[("fg", "fg")] = G["Frankfort"]["Gulfport"]['weight']

print("Q matrix:\n")

print("\t ab \t ac \t ad \t ae \t af \t ag \t bc \t ... \t fg")
print("ab [\t",Q[("ab", "ab")], "\t \t \t \t \t \t \t \t \t]")
print("ac [\t \t",Q[("ac", "ac")],"\t \t \t \t \t \t \t \t]")
print("ad [\t \t \t",Q[("ad","ad")],"\t \t \t \t \t \t \t]")
print("ae [\t \t \t \t",Q[("ae","ae")],"\t \t \t \t \t \t]")
print("af [\t \t \t \t \t",Q[("af","af")],"\t \t \t \t \t]")
print("ag [\t \t \t \t \t \t",Q[("ag","ag")]," \t \t \t \t]")
print("bc [\t \t \t \t \t \t \t",Q[("bc","bc")],"\t \t \t]")
print("...[\t \t \t \t \t \t \t \t... \t \t]")

print("fg [\t \t \t \t \t \t \t \t \t",Q[("fg","fg")]," \t]")


# ### Constraints
# 
# Our constraints in this problem are to visit each city exactly once, and end where we started.
# 
# For our constraints, consider city $A$ first.  The city needs to appear exactly twice in our list of variables: one leg into the city and one leg out of the city.  This translates to the following requirement: 
# 
# $$ab + ac + ad + ae + af + ag = 2.$$ 
# 
# To make this constraint "QUBO appropriate", we need to write it as an expression that is true at the smallest value.  For constraints of the type "exactly 2", we move everything to one side and square it.
# 
# $$(ab + ac + ad + ae + af + ag - 2)^2$$
# 
# You can picture this in two dimensions as $y = (x-2)^2$, which has a nice parabola shape with a distinct minimum value at $x=2$.
# 
# When we multiply our constraint out we get the following expression.
# 
# $$\begin{eqnarray*}
#     & & ab^2 + ac^2 + ad^2 + ae^2 + af^2 + ag^2 +4 \\
#     &+& abac + abad + abae + abaf + abag -2ab \\
#     &+& acab + acad + acae + acaf + acag -2ac \\
#     &+& \cdots \\
#     &+& -2ab -2ac-2ad-2ae-2af-2ag
# \end{eqnarray*}$$
# 
# Remembering that we can remove constants, and that a binary variable $x$ always has $x^2=x$, we can simplify more to get:
# 
# $$\begin{eqnarray*}
#     & & -3ab -3ac -3ad -3ae -3af -3ag \\
#     & + & 2abac + 2abad + 2abae + 2abaf + 2abag \\
#     & + & 2acad + 2acae + 2acaf + 2acag + 2adae \\
#     & + & 2adaf + 2adag + 2aeaf + 2aeag + 2afag
# \end{eqnarray*}$$
# 
# When we multiply this by the Lagrange parameter $\gamma$ (gamma), we can update the QUBO dictionary entries below.
# 
# The first cell adds the coefficients from our constraint that Albuquerque must appear twice in our leg variables.

# In[ ]:


from itertools import combinations

for i in ['b','c','d','e','f','g']:
    Q[('a'+i,'a'+i)] += -3*gamma
    
for pair in list(combinations(['b','c','d','e','f','g'], 2)):
    Q[('a'+pair[0],'a'+pair[1])] = 2*gamma

print("Q matrix:\n")

print("\t ab \t ac \t ad \t ae \t af \t ag \t bc \t ... \t fg")
print("ab [\t",Q[("ab", "ab")],"\t", Q[("ab","ac")],"\t",Q[("ab","ad")], "\t",Q[("ab","ae")],"\t",Q[("ab","af")],"\t",Q[("ab","ag")], "\t \t \t \t]")
print("ac [\t \t",Q[("ac", "ac")],"\t",Q[("ac","ad")],"\t",Q[("ac","ae")],"\t",Q[("ac","af")],"\t",Q[("ac","ag")],"\t \t \t \t]")
print("ad [\t \t \t",Q[("ad","ad")],"\t",Q[("ad","ae")],"\t",Q[("ad","af")],"\t",Q[("ad","ag")],"\t \t \t \t]")
print("ae [\t \t \t \t",Q[("ae","ae")],"\t",Q[("ae","af")],"\t",Q[("ae","ag")],"\t \t \t \t]")
print("af [\t \t \t \t \t",Q[("af","af")],"\t", Q[("af","ag")],"\t \t \t \t]")
print("ag [\t \t \t \t \t \t",Q[("ag","ag")],"\t \t \t \t]")
print("bc [\t \t \t \t \t \t \t",Q[("bc","bc")],"\t \t \t]")
print("...[\t \t \t \t \t \t \t \t... \t \t]")

print("fg [\t \t \t \t \t \t \t \t \t",Q[("fg","fg")]," \t]") 


# We use the same method to implement the constraints 
#     -  "Visit Boston exactly once in our cycle"
#     -  "Visit Charlotte exactly once in our cycle"
#     -  "Visit Detroit exactly once in our cycle"
#     -  "Visit Evanston exactly once in our cycle"
#     -  "Visit Frankfort exactly once in our cycle"
#     -  "Visit Gulfport exactly once in our cycle"

# In[ ]:


## We need to make sure our variable names are letters in alphabetical order:  ab and not ba!
#  We can do this using ''.join(sorted(...))

# Visit Boston exactly once in our cycle
for i in ['a','c','d','e','f','g']:
    Q[(''.join(sorted('b'+i)),''.join(sorted('b'+i)))] += -3*gamma
for pair in list(combinations(['a','c','d','e','f','g'], 2)):
    Q[(''.join(sorted('b'+pair[0])),''.join(sorted('b'+pair[1])))] = 2*gamma
    
# Visit Charlotte exactly once in our cycle
for i in ['a','b','d','e','f','g']:
    Q[(''.join(sorted('c'+i)),''.join(sorted('c'+i)))] += -3*gamma
for pair in list(combinations(['a','b','d','e','f','g'], 2)):
    Q[(''.join(sorted('c'+pair[0])),''.join(sorted('c'+pair[1])))] = 2*gamma
    
# Visit Detroit exactly once in our cycle
for i in ['a','b','c','e','f','g']:
    Q[(''.join(sorted('d'+i)),''.join(sorted('d'+i)))] += -3*gamma
for pair in list(combinations(['a','b','c','e','f','g'], 2)):
    Q[(''.join(sorted('d'+pair[0])),''.join(sorted('d'+pair[1])))] = 2*gamma
    
# Visit Evanston exactly once in our cycle
for i in ['a','b','c','d','f','g']:
    Q[(''.join(sorted('e'+i)),''.join(sorted('e'+i)))] += -3*gamma
for pair in list(combinations(['a','b','c','d','f','g'], 2)):
    Q[(''.join(sorted('e'+pair[0])),''.join(sorted('e'+pair[1])))] = 2*gamma
    
# Visit Frankfort exactly once in our cycle
for i in ['a','b','c','d','e','g']:
    Q[(''.join(sorted('f'+i)),''.join(sorted('f'+i)))] += -3*gamma
for pair in list(combinations(['a','b','c','d','e','g'], 2)):
    Q[(''.join(sorted('f'+pair[0])),''.join(sorted('f'+pair[1])))] = 2*gamma
    
# Visit Gulfport exactly once in our cycle
for i in ['a','b','c','d','e','f']:
    Q[(''.join(sorted('g'+i)),''.join(sorted('g'+i)))] += -3*gamma
for pair in list(combinations(['a','b','c','d','e','f'], 2)):
    Q[(''.join(sorted('g'+pair[0])),''.join(sorted('g'+pair[1])))] = 2*gamma

# Print our Q matrix
print("\t ab \t ac \t ad \t ae \t af \t ag \t bc \t ... \t fg")
print("ab [\t",'\t'.join([str(Q[("ab",'a'+k)]) for k in ['b','c','d','e','f','g']]),"\t",Q[("ab","bc")],"\t...\t",Q[("ab","fg")],"\t]")
print("ac [\t",'\t'.join([str(Q[("ac",'a'+k)]) for k in ['b','c','d','e','f','g']]),"\t",Q[("ac","bc")],"\t...\t",Q[("ac","fg")],"\t]")
print("ad [\t",'\t'.join([str(Q[("ad",'a'+k)]) for k in ['b','c','d','e','f','g']]),"\t",Q[("ad","bc")],"\t...\t",Q[("ad","fg")],"\t]")
print("ae [\t",'\t'.join([str(Q[("ae",'a'+k)]) for k in ['b','c','d','e','f','g']]),"\t",Q[("ae","bc")],"\t...\t",Q[("ae","fg")],"\t]")
print("af [\t",'\t'.join([str(Q[("af",'a'+k)]) for k in ['b','c','d','e','f','g']]),"\t",Q[("af","bc")],"\t...\t",Q[("af","fg")],"\t]")
print("ag [\t",'\t'.join([str(Q[("ag",'a'+k)]) for k in ['b','c','d','e','f','g']]),"\t",Q[("ag","bc")],"\t...\t",Q[("ag","fg")],"\t]")
print("bc [\t",'\t'.join([str(Q[("bc",'a'+k)]) for k in ['b','c','d','e','f','g']]),"\t",Q[("bc","bc")],"\t...\t",Q[("bc","fg")],"\t]")
print("...[\t... \t... \t... \t... \t... \t... \t... \t... \t... \t]")
print("fg [\t",'\t'.join([str(Q[("fg",'a'+k)]) for k in ['b','c','d','e','f','g']]),"\t",Q[("fg","bc")],"\t...\t",Q[("fg","fg")],"\t]")


# ## Sending our QUBO to the QPU
# 
# Now that we have created our QUBO dictionary that represents the coefficients in our equation we can embed the problem and send it to the QPU. We store the results in the variable "response".

# In[ ]:


from dwave.system.samplers import DWaveSampler           # Library to interact with the QPU
from dwave.system.composites import EmbeddingComposite   # Library to embed our problem onto the QPU physical graph
response = EmbeddingComposite(DWaveSampler()).sample_qubo(Q, chain_strength=chainstrength, num_reads=numruns, label='Notebook - 7-City Traveling Salesman')   # Some accounts need to replace this line with the next:
# response = EmbeddingComposite(DWaveSampler(token='my_token', solver='solver_name')).sample_qubo(Q, chain_strength=chainstrength, num_reads=numruns) 

print("QPU call complete using", response.info['timing']['qpu_access_time'], "microseconds of QPU time.")


# ## Post-Processing
# 
# Next, we need to interpret the results that the machine returns to us.  We can tie the values returned to our original binary variables (legs), and print them out to look at and understand them.  We will print out the energy for the solution found, followed by the legs used.  We will only print out the first 10 results.
# 
# Notice that the QPU returns the lowest energy response first.  
# 
# If the answers don't seem very good, you can change your values for $\textbf{gamma}$, $\textbf{chainstrength}$, and $\textbf{numruns}$ and re-run the program.

# In[ ]:


R = iter(response)
E = iter(response.data())
ndx=0
tours={}
Energy={}
for line in response:
    ndx=ndx+1
    sample = next(R)
    Energy[ndx]=(next(E).energy)
    tours[ndx]=([node for node in sample if sample[node] > 0]) 
    
for i in range(10):
    print(Energy[i+1], tours[i+1])


# ### How do we know if our answer is any good?
# 
# Since the QPU is probabilistic, without running an exact solver as well we have no way of knowing the "quality" of our answer without some other method.
# 
# One method is $\textbf{visualization}$.  For a problem like TSP, we can visualize the best solutions found on a map to determine the quality of the solution.  We won't know if it's the best possible, but we can have an idea if might be a good route.

# In[ ]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import pause
import pylab
get_ipython().run_line_magic('matplotlib', 'inline')

Path = nx.Graph()

for solution in range(1,10):
    plt.figure(figsize=(20,10))
    Path.clear()
    print(Energy[solution])
    print(tours[solution])
    for ndx in range(len(tours[solution])):
        nx.add_path(Path, tours[solution][ndx])
    Path = nx.relabel_nodes(Path, map, copy=True)

    for i in range(7):
        coord[i]=(Lat[i],Lon[i])
        Path.add_node(Cities[i],pos=coord[i],label=Cities[i])

    nx.draw_networkx_nodes(Path, nx.get_node_attributes(Path, 'pos'), node_size=200)
    nx.draw_networkx_labels(Path, nx.get_node_attributes(Path, 'pos'),font_size=18, font_weight='bold')
    nx.draw_networkx_edges(Path, nx.get_node_attributes(Path, 'pos'), edge_color='b')
    plt.axis('off')
    plt.show()


# ## What's Next?
# 
# Now that you have visualized your routes, you may have some routes that are two separate pieces.  This is what is called $\textbf{subloops}$.  Technically we have satisfied the constraints that we gave the QPU through our QUBO:  each city is visited exactly once, but we have two separate shorter routes.
# 
# To fix this problem, take a look at the 48-city Traveling Salesman Notebook where we look at a different approach to formulating a QUBO for the same problem.

# Copyright &copy; D-Wave Systems Inc.
