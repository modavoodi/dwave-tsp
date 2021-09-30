
# coding: utf-8

# # 48-City Traveling Salesman Problem
# 
# Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin city?
# 
# ## Look at our Data
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

# Read in our cities lat/long coordinates from a file
N = 48
with open('TSP_48_capitals.txt', "r") as myfile:
    city_text = myfile.readlines()
    myfile.close()
cities = [',']*N
states = [',']*N
lats=[]
longs=[]
for i in city_text:
    index, state, city,lat,lon = i.split(',')
    cities[int(index)] = city.rstrip()
    states[int(index)] = state
    lats.append(float(lat))
    longs.append(float(lon))

# initialize figure
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
output_notebook()
p = figure(title="Find shortest route that visits each city", 
           toolbar_location="left", plot_width=550, plot_height=350)

# Draw state lines
p.patches(state_xs, state_ys, fill_alpha=0.0,
    line_color='blue', line_width=1.5)

# Place markers for each city
p.circle(longs, lats, size=10, color='red', alpha=1)

# Display the image
show(p)


# ## Setting up our Graph
# 
# First, we set up our problem as a graph.  Our nodes are the cities our salesman needs to visit, and our edges are the roads between any pair of cities.

# In[ ]:


# Open the input file containing inter-city distances
import re
fn = "TSP_48_state_capitals.txt"

with open(fn, "r") as myfile:
    distance_text = myfile.readlines()

# Extract the distances from the input file
D = [[0 for z in range(N)] for y in range(N)]
for i in distance_text:
    if re.search("^between", i):
        m = re.search("^between_(\d+)_(\d+) = (\d+)", i)
        citya = int(m.group(1))
        cityb = int(m.group(2))
        D[citya][cityb] = D[cityb][citya] = int(m.group(3))
    
print("Read in data for", len(D),"cities.")


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


# Tunable parameters. 
gamma = 8500
chainstrength = 4500
numruns = 100
print("Tunable parameters: \n\tGamma: \t\t\t",gamma,"\n\tChain Strength: \t",chainstrength,"\n\tNumber of runs: \t",numruns)


# ## Building our QUBO
# 
# We start by creating an empty Q matrix.

# In[ ]:


Q = {}
for i in range(N*N):
    for j in range(N*N):
        Q[(i,j)] = 0

print("Q matrix with", len(Q), "entries created.")


# Since our variables have two indices, we'll provide a helper function help us assign $x_{a,b}$ to a specific row/column index in our Q matrix.

# In[ ]:


# Function to compute index in Q for variable x_(a,b)
def x(a, b):
    return (a)*N+(b)


# ### Row Constraints
# 
# The next block sets the constraint that each row has exactly one 1 in our permutation matrix.
# 
# For row 1, this constraint looks like $$\gamma \left(-1 + \sum_{j=1}^{48} x_{1,j} \right).$$  When we simplify, we get $$\sum_{j=1}^{48} \left(-\gamma x_{1,j}\right) + \sum_{j=1}^{48}\sum_{k=j+1}^{48} \left(2\gamma x_{1,j}x_{1,k}\right).$$

# In[ ]:


for v in range(N):
    for j in range(N):
        Q[(x(v,j), x(v,j))] += -1*gamma
        for k in range(j+1, N):
            Q[(x(v,j), x(v,k))] += 2*gamma
            
print("Added",N,"row constraints to Q matrix.")


# ### Column Constraints
# 
# The next block sets the constraint that each column has exactly one 1 in our permutation matrix.
# 
# For column 1, this constraint looks like $$\gamma \left(-1 + \sum_{v=1}^{48} x_{v,1} \right).$$  When we simplify, we get $$\sum_{v=1}^{48} \left(-\gamma x_{v,1}\right) + \sum_{v=1}^{48}\sum_{w=v+1}^{48} \left(2 \gamma x_{v,1}x_{w,1}\right).$$

# In[ ]:


for j in range(N):
    for v in range(N):
        Q[(x(v,j), x(v,j))] += -1*gamma
        for w in range(v+1,N):
            Q[(x(v,j), x(w,j))] += 2*gamma
            
print("Added",N,"column constraints to Q matrix.")


# ### Objective Function
# 
# Our objective is to minimize the distanced travelled.  
# 
# The distance we travel from city $u$ to city $v$ in stops $2$ and $3$ is $D(u,v)x_{u,2}x_{v,3}$. This adds $D(u,v)$ to our total distance if we visit city $u$ in stop 2 and city $v$ in stop 3, and adds 0 to our total distance otherwise.
# 
# So, for every pair of cities $u$ and $v$, we add $\sum_{j=1}^{48} D(u,v)x_{u,j}x_{v,j+1}$ to add the distance travelled from $u$ to $v$ (directly) in our route.
# 
# We need to add this for every choice of $u$ and $v$ (and in both directions).

# In[ ]:


for u in range(N):
    for v in range(N):
        if u!=v:
            for j in range(N):
                Q[(x(u,j), x(v,(j+1)%N))] += D[u][v]
                
print("Objective function added.")


# ## Running QBSolv Classically
# 
# Qbsolv is a problem decomposition tool that can either run offline (classically) or in a hybrid manner (classical-QPU combination).  Qbsolv is based off of the paper "A Multilevel Algorithm for Large Unconstrained Binary Quadratic Optimization", Wang, Lu, Glover, and Hao (2012).
# 
# First we will run our problem with QBSolv classically.

# In[ ]:


import time
from dwave_qbsolv import QBSolv

start = time.time()
resp = QBSolv().sample_qubo(Q)
end = time.time()
time_CPU = end - start

print("QBSolv (offline) sampling complete using",time_CPU,"seconds.")


# ### Understanding the Results
# 
# Once we run QBSolv, we need to collect and report back the best answer found.  Here we list off the lowest energy solution found and the total mileage required for this route. If you wish to see the cities in order of the route, uncomment out the code at the end of the next cell block.

# In[ ]:


# First solution is the lowest energy solution found
sample = next(iter(resp))

# Display energy for best solution found
print('Energy: ', next(iter(resp.data())).energy)

# Print route for solution found
route = [-1]*N
for node in sample:
    if sample[node]>0:
        j = node%N
        v = (node-j)/N
        if route[j]!=-1:
            print('Stop '+str(i)+' used more than once.\n')
        route[j] = int(v)
        
# Compute and display total mileage
mileage = 0
for i in range(N):
    mileage+=D[route[i]][route[(i+1)%N]]
mileage_CPU = mileage
print('Mileage: ', mileage_CPU)
        
##--- Uncomment below to print out route ---##
# print('\nRoute:\n')
# for i in range(N):
#     if route[i]!=-1:
#         print(str(i) + ':  ' +cities[route[i]]+ ',' + states[route[i]] + '\n')  
#     else:
#         print(str(i) + ':  No city assigned.\n')


# ### Checking Our Answer
# 
# Is this answer valid?  Here we provide a few checks.
# 
# First, if every city appears exactly once in our list, then our route list will consist of the numbers $0, 1, 2, \ldots , N-1$ in some order, which add up to $N(N-1)/2$.  If this sum is not correct, our route is invalid.
# 
# Second, we check to see if every stop has a city assigned.  If not, we print a message to the user to make them aware.
# 
# An additional check that you might want to implement would check if any city is assigned to more than one stop.

# In[ ]:


flag = 0

if sum(route)!=N*(N-1)/2:
    flag = 1
    print('Route invalid.\n')

for i in range(N):
    if route[i]==-1:
        flag = 1
        print('Stop '+str(i)+' has no city assigned.')
        
if flag==0:
    print("Route valid.")


# ## Running QBSolv with the QPU
# 
# Next we will run our problem using QBSolv with the QPU.
# 
# Because the QPU has a fixed number of qubits, we need to limit the size of our subproblems so that each subproblem can be embedded onto the chip.  The largest complete graph that can be embedded on the chip has a little over 60 nodes, so we will limit our subproblem size to 60.
# 
# We will compare using `EmbeddingComposite` (embed each subproblem on the chip as its needed) versus `FixedEmbeddingComposite` (precompute an embedding of a complete graph of 60 nodes on the chip for reuse for every subproblem) to illustrate two different approachs to running this problem.
# 
# ### EmbeddingComposite

# In[ ]:


from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler

sampler = DWaveSampler()  # Some accounts need to replace this line with the next:
# sampler = EmbeddingComposite(DWaveSampler(token = 'my_token', solver=dict(name='solver_name')))

print("Connected to QPU.")


# In the next cell we will call QBSolv to sample our QUBO.  We will use the following QBSolv parameters.
#     -  solver_limit:  Size of the subproblems.
#     -  timeout:  Limits the amount of time that QBSolv can run.
#     -  num_repeats:  Number of times the QBSolv algorithm will re-partition the problem.
#     
# We will use the following QPU parameters.
#     -  num_reads:  Number of samples we wish to obtain for each subproblem from the QPU.
#     -  chain_strength:  The chain strength to use for our embedding.
#     
# You may get an error that no embedding was found.  This means that our heuristic embedding algorithm was not able to quickly find an embedding for a subproblem, and you need to run the cell again.

# In[ ]:


start = time.time()
resp = QBSolv().sample_qubo(Q, solver=EmbeddingComposite(sampler), solver_limit=60, timeout=30, num_repeats=1, num_reads=numruns, chain_strength=chainstrength)
end = time.time()
time_QPU = end-start

print(time_QPU,"seconds of wall-clock time.")


# Let's look at our top answer.  We will print out the mileage corresponding to the lowest energy solution found, as well as an error-message if the route is invalid.

# In[ ]:


# First solution is the lowest energy solution found
sample = next(iter(resp))

# Display energy for best solution found
print('Energy: ', next(iter(resp.data())).energy)

# Print route for solution found
route = [-1]*N
for node in sample:
    if sample[node]>0:
        j = node%N
        v = (node-j)/N
        route[j] = int(v)   
        
# Compute and display total mileage
mileage = 0
for i in range(N):
    mileage+=D[route[i]][route[(i+1)%N]]
mileage_QPU = mileage
print('Mileage: ', mileage_QPU)

if sum(route)!=N*(N-1)/2:
    print('Route invalid.\n')

for i in range(N):
    if route[i]==-1:
        print('Stop '+str(i)+' has no city assigned.')


# ### FixedEmbeddingComposite
# 
# Now we'll run QBSolv with the QPU using `FixedEmbeddingComposite`.  This will ensure that we never have embedding issues.
# 
# Our first step is to pre-compute an embedding for a complete graph on 60 nodes onto the chip topology.

# In[ ]:


import networkx as nx
import minorminer

G = nx.complete_graph(60)
embedding = minorminer.find_embedding(G.edges, sampler.edgelist)
print("Embedding found.")


# Now that we have computed our embedding, we pass our sampler (the QPU) and our embedding together into  `FixedEmbeddingComposite`.  We will use the same parameters as in the previous QBSolv `sample_qubo` call.

# In[ ]:


from dwave.system.composites import FixedEmbeddingComposite

start = time.time()
resp = QBSolv().sample_qubo(Q, solver=FixedEmbeddingComposite(sampler, embedding), solver_limit=60, timeout=30, num_repeats=1, num_reads=numruns, chain_strength=chainstrength)
end = time.time()
time_QPU_Fixed = end-start

print(time_QPU_Fixed,"seconds of wall-clock time.")


# Let's look at our top answer.  We will print out the mileage corresponding to the lowest energy solution found, as well as an error-message if the route is invalid.

# In[ ]:


# First solution is the lowest energy solution found
sample = next(iter(resp))

# Display energy for best solution found
print('Energy: ', next(iter(resp.data())).energy)

# Print route for solution found
route = [-1]*N
for node in sample:
    if sample[node]>0:
        j = node%N
        v = (node-j)/N
        route[j] = int(v)   
        
# Compute and display total mileage
mileage = 0
for i in range(N):
    mileage+=D[route[i]][route[(i+1)%N]]
mileage_QPU_Fixed = mileage
print('Mileage: ', mileage_QPU_Fixed)

if sum(route)!=N*(N-1)/2:
    print('Route invalid.\n')

for i in range(N):
    if route[i]==-1:
        print('Stop '+str(i)+' has no city assigned.')


# ## Visualization
# 
# This last block visualizes the best route found using `FixedEmbeddingComposite`.  How does it look?

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

Path = nx.Graph()
coord={}
coord[route[0]]=(longs[route[0]],lats[route[0]])
Path.add_node(cities[route[0]],pos=coord[route[0]],label=cities[route[0]])

for i in range(N-1):
    e=(cities[route[i]],cities[route[i+1]])
    Path.add_edge(*e)
    coord[route[i+1]]=(longs[route[i+1]],lats[route[i+1]])
    Path.add_node(cities[route[i+1]],pos=coord[route[i+1]],label=cities[route[i+1]])

e=(cities[route[N-1]],cities[route[0]])
Path.add_edge(*e)
    
fig, ax = plt.subplots(figsize=(120,60))
margin=0.15
fig.subplots_adjust(margin, margin, 1.-margin, 1.-margin)
ax.axis('equal')
nx.draw(Path, nx.get_node_attributes(Path, 'pos'), with_labels=True, width=10, edge_color='b', node_size=200,font_size=72,font_weight='bold', ax=ax)
plt.show() 


# ## Summary
# 
# To summarize our results using QBSolv classically, with `EmbeddingComposite`, and with `FixedEmbeddingComposite`, we provide a table of the results and wall-clock run times.  Remember that you may want to adjust the tunable parameters `gamma`, `chainstrength`, and `numruns` above to improve performance.

# In[ ]:


print("QBSolv \t\t\t| Mileage \t| Run Time (s)")
print("--------------------------------------------------------------------")
print("Classical \t\t| ",mileage_CPU,"\t| ",time_CPU)
print("EmbeddingComposite \t| ",mileage_QPU,"\t| ",time_QPU)
print("FixedEmbeddingComposite | ",mileage_QPU_Fixed,"\t| ",time_QPU_Fixed)


# Copyright &copy; D-Wave Systems Inc.
