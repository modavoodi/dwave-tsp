
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

# Read in the lat/long coordinates for cities from a file
import pandas as pd
city_info = pd.read_csv('TSP_48_capitals_Hybrid.txt')

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
p.circle(city_info['Long'], city_info['Lat'], size=10, color='red', alpha=1)

# Display the image
show(p)


# ## Setting up our Graph
# 
# First, we set up our problem as a graph.  Our nodes are the cities our salesman needs to visit, and our edges are the roads between any pair of cities.

# In[ ]:


# Open the input file containing inter-city distances
import re
fn = "TSP_48_state_capitals.txt"
N=48

with open(fn, "r") as myfile:
    distance_text = myfile.readlines()

# Extract the distances from the input file
D = [[0 for z in range(N)] for y in range(N)]
for line in distance_text:
    m = re.match("^between_(\d+)_(\d+) = (\d+)", line)
    if m:
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
# 
# Normally, a good way to do this would be to use the `defaultdict` function in Python's `collections` package.  However, to explore the density of our matrix (how many 0 entries we have), we will intialize a full matrix with 0's and then remove any 0 entries later.  Note that this method is slower since we are creating many more entries than our required and then deleting entries, rather than only creating the matrix entries we need.

# In[ ]:


Q = {}
for i in range(N*N):
    for j in range(N*N):
        Q[(i,j)] = 0

print("Q matrix with", len(Q), "entries created.")


# Since our variables have two indices, we'll provide a helper function help us assign $x_{a,b}$ to a specific row/column index in our Q matrix.

# In[ ]:


# Function to compute index in Q for variable x_(a,b)
def variable_to_index(a, b):
    return a*N+b

# Function to compute variable given index
def index_to_variable(index):
    j = index%N
    return [j,(index-j)/N]


# ### Row Constraints
# 
# The next block sets the constraint that each row has exactly one 1 in our permutation matrix.
# 
# For row 1, this constraint looks like $$\gamma \left(-1 + \sum_{j=1}^{48} x_{1,j} \right).$$  When we simplify, we get $$\sum_{j=1}^{48} \left(-\gamma x_{1,j}\right) + \sum_{j=1}^{48}\sum_{k=j+1}^{48} \left(2\gamma x_{1,j}x_{1,k}\right).$$

# In[ ]:


for v in range(N):
    for j in range(N):
        Q[(variable_to_index(v,j), variable_to_index(v,j))] += -1*gamma
        for k in range(j+1, N):
            Q[(variable_to_index(v,j), variable_to_index(v,k))] += 2*gamma
            
print("Added",N,"row constraints to Q matrix.")


# ### Column Constraints
# 
# The next block sets the constraint that each column has exactly one 1 in our permutation matrix.
# 
# For column 1, this constraint looks like $$\gamma \left(-1 + \sum_{v=1}^{48} x_{v,1} \right).$$  When we simplify, we get $$\sum_{v=1}^{48} \left(-\gamma x_{v,1}\right) + \sum_{v=1}^{48}\sum_{w=v+1}^{48} \left(2 \gamma x_{v,1}x_{w,1}\right).$$

# In[ ]:


for j in range(N):
    for v in range(N):
        Q[(variable_to_index(v,j), variable_to_index(v,j))] += -1*gamma
        for w in range(v+1,N):
            Q[(variable_to_index(v,j), variable_to_index(w,j))] += 2*gamma
            
print("Added",N,"column constraints to Q matrix.")


# ### Objective Function
# 
# Our objective is to minimize the distance travelled.  
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
                Q[(variable_to_index(u,j), variable_to_index(v,(j+1)%N))] += D[u][v]
                
print("Objective function added.")


# Since we started out with a matrix full of zeros, we probably have a lot of entries that are still 0.  Removing them will make our problem easier to decompose and run on the QPU.

# In[ ]:


Q = {k:v for k,v in Q.items() if v!=0}
print("Q matrix reduced to", len(Q),"entries.")


# ## Running dwave-hybrid's `KerberosSampler`
# 
# The dwave-hybrid package comes with a ready-to-go sampler that runs using the same syntax as all of the dimod samplers that you are familiar with.  Let's run this first to get a baseline to get started, then explore what a workflow looks like and how we can improve our results.

# In[ ]:


from hybrid import KerberosSampler
from dwave.system.samplers import DWaveSampler

sampler = KerberosSampler()
resp = sampler.sample_qubo(Q)

print("KerberosSampler call complete.")


# ### Understanding the Results
# 
# Once we run `KerberosSampler`, we need to collect and report back the best answer found.  Here we build a function to do some initial tests of our best sample returned.  When it runs, it will list off the lowest energy solution found and the total mileage required for this route. If you wish to see the cities in order of the route, uncomment out the code at the end of the next cell block.

# In[ ]:


# The function below will compute the route from the best sample found
def compute_route(sample):
    # First solution is the lowest energy solution found
    route = [-1]*N
    for node in sample:
        if sample[node]>0:
            [j,v] = index_to_variable(node)
            if route[j]!=-1:
                print('Stop ',i,' used more than once.\n')
            route[j] = int(v)
    return route

# The function below will print out the corresponding mileage for best sample/route found
def print_result(route):
    # Compute and display total mileage
    mileage = 0
    for i in range(N):
        mileage+=D[route[i]][route[(i+1)%N]]
    print('Mileage: ', mileage)
    return mileage

# The function below will print out the full route
def print_route(resp):
    print('\nRoute:\n')
    for i in range(N):
        if route[i]!=-1:
            print(i,':  ',city_info.iloc[route[i],1],',',city_info.iloc[route[i],0],'\n')  
        else:
            print(i,':  No city assigned.\n')

sample = next(iter(resp))
route = compute_route(sample)

# Display energy and mileage for best solution found
print('Energy: ', next(iter(resp.data())).energy)
mileage_KS = print_result(route)


# ### Checking Our Answer
# 
# Is this answer valid?  Here we provide a few checks.
# 
# First, if every city appears exactly once in our list, then our route list will consist of the numbers $0, 1, 2, \ldots , N-1$ in some order, which add up to $N(N-1)/2$.  If this sum is not correct, our route is invalid.
# 
# Second, we check to see if every stop has a city assigned.  If not, we print a message to the user to make them aware.
# 
# An additional check that you might want to implement would check if any city is assigned to more than one stop.  Note that we checked this quickly as we constructed our route from our samples in the function `compute_route`.

# In[ ]:


def check_route(route):
    route_invalid = 0

    if sum(route)!=N*(N-1)/2:
        route_invalid = 1
        print('Route invalid.\n')

    for i in range(N):
        if route[i]==-1:
            route_invalid = 1
            print('Stop '+str(i)+' has no city assigned.')

    if route_invalid==0:
        print("Route valid.")
        
check_route(route)


# ## Setting up a dwave-hybrid workflow (hybrid-sampler)
# 
# Now let's take a look at what's going on under the hood in `KerberosSampler`, and explore what a dwave-hybrid workflow looks like.
# 
# First, we can find the source code for `KerberosSampler` at https://github.com/dwavesystems/dwave-hybrid/blob/0.3.1/hybrid/reference/kerberos.py#L91.  Lines 91-107 give us a general layout of how the `KerberosSampler` workflow is built.
# 
# Let's start with the Kerberos workflow to begin with, and see how we can create more intricate, tuned workflows to improve our results.  The workflow in the next cell can be described with a visual flow chart.
# 
# ![title](racing_branches_1.png)

# In[ ]:


# Set up our hybrid workflow
import hybrid
iteration = hybrid.RacingBranches(
    hybrid.InterruptableTabuSampler(timeout=1000),
    hybrid.EnergyImpactDecomposer(size=50, rolling=True, traversal='bfs')
    | hybrid.QPUSubproblemAutoEmbeddingSampler()
    | hybrid.SplatComposer()
) | hybrid.ArgMin()
workflow = hybrid.Loop(iteration, convergence=3)
dimod_sampler = hybrid.HybridSampler(workflow)

resp = dimod_sampler.sample_qubo(Q)

print("Hybrid workflow call complete.\n")

sample = next(iter(resp))
route = compute_route(sample)

# Display energy for best solution found
print('Energy: ', next(iter(resp.data())).energy)
mileage_workflow1 = print_result(route)


# ### Using Runnables
# 
# In building dwave-hybrid, we created a class of objects called "runnables".  Runnables are like functions that take as input some initial state to our problem, run the workflow, and return an output state.  When we use runnables, we are able to execute our branches asynchronously, since a future object is created.  This object will contain the results of the computation once it is complete.  A workflow can be composed of several runnable objects, and the asynchronous nature of these objects allows us to execute many branches at the same time.
# 
# In the next cell, we repeat the same workflow we just used but with the syntax corresponding to these runnable objects.  Additionally, you will notice that we convert our QUBO over to a binary quadratic model (BQM) object using `dimod`.  The class BQM contains both QUBO and Ising models, and using the BQM object allows us to switch between the two models.  For more information on BQMs, see the documentation (https://docs.ocean.dwavesys.com/projects/dimod/en/latest/introduction.html).

# In[ ]:


iteration = hybrid.RacingBranches(
    hybrid.InterruptableTabuSampler(timeout=1000),
    hybrid.EnergyImpactDecomposer(size=50, rolling=True, traversal='bfs')
    | hybrid.QPUSubproblemAutoEmbeddingSampler()
    | hybrid.SplatComposer()
) | hybrid.ArgMin()
workflow = hybrid.Loop(iteration, convergence=3)

import dimod
model = dimod.BinaryQuadraticModel.from_qubo(Q)
init_state = hybrid.State.from_sample(hybrid.min_sample(model), model)
resp = workflow.run(init_state).result()

print("\nHybrid workflow call complete.\n")

# First solution is the lowest energy solution found
sample = resp.samples.first.sample

# Display energy for best solution found
print('Energy: ', resp.samples.first.energy)

route = compute_route(sample)
mileage_workflow2 = print_result(route)


# ## Visualization
# 
# This next block visualizes the last route found in our last workflow.  How does it look?

# In[ ]:


import matplotlib.pyplot as plt
import networkx as nx
get_ipython().run_line_magic('matplotlib', 'inline')

path = nx.Graph()
coord = (city_info.iloc[route[0],3], city_info.iloc[route[0],2])
path.add_node(city_info.iloc[route[0],1], pos=coord)

for i in range(N-1):
    e = (city_info.iloc[route[i],1], city_info.iloc[route[i+1],1])
    path.add_edge(*e)
    coord = (city_info.iloc[route[i+1],3], city_info.iloc[route[i+1],2])
    path.add_node(city_info.iloc[route[i+1],1], pos=coord)

e = (city_info.iloc[route[N-1],1], city_info.iloc[route[0],1])
path.add_edge(*e)
    
fig, ax = plt.subplots(figsize=(120,60))
margin=0.15
fig.subplots_adjust(margin, margin, 1.-margin, 1.-margin)
ax.axis('equal')
nx.draw(path, nx.get_node_attributes(path, 'pos'), with_labels=True, width=10, edge_color='b', node_size=200,font_size=72,font_weight='bold', ax=ax)
plt.show() 


# ## Summary
# 
# To summarize our results using dwave-hybrid, we provide a table of the mileage results found in this notebook.  Recall that each different method we used to run dwave-hybrid implemented the same workflow, so we should expect to see relatively similar results.  To improve performance, you may want to adjust the tunable parameters `gamma`, `chainstrength`, and `numruns`, as well as exploring more complex workflows.

# In[ ]:


print("Workflow \t\t| Mileage")
print("--------------------------------------------------------------------")
print("KerberosSampler \t| ",mileage_KS)
print("Dimod-Cast Workflow \t| ",mileage_workflow1)
print("Runnables Workflow \t| ",mileage_workflow2)


# Copyright &copy; D-Wave Systems Inc.
