# YOUR CODE HERE
import sys
from collections import defaultdict, deque

def solve():
    N = int(sys.stdin.readline())
    S = [sys.stdin.readline().strip() for _ in range(N)]

    # Adjacency list for forward edges (X -> Y for product XY)
    adj = defaultdict(list)
    # Adjacency list for backward edges (Y <- X for product XY), used for finding weakly connected components
    rev_adj = defaultdict(list)
    
    # Store in-degree and out-degree for each vertex
    indeg = defaultdict(int)
    outdeg = defaultdict(int)
    
    # Keep track of all vertices that appear in any product name
    vertices = set() 

    # Build the graph and compute degrees
    for s_i in S:
        # Each product name S_i = XY corresponds to a directed edge X -> Y
        u, v = s_i[0], s_i[1]
        adj[u].append(v)
        rev_adj[v].append(u)
        
        # Increment out-degree of start vertex u
        outdeg[u] += 1
        # Increment in-degree of end vertex v
        indeg[v] += 1
        
        # Add both vertices to the set of vertices
        vertices.add(u)
        vertices.add(v)

    # Identify source vertices: those with in-degree 0 and out-degree > 0
    sources = set()
    # Identify sink vertices: those with out-degree 0 and in-degree > 0
    sinks = set()

    # Iterate over all vertices involved in edges to find global sources and sinks
    for v in vertices: 
        if indeg[v] == 0 and outdeg[v] > 0:
            sources.add(v)
        if outdeg[v] == 0 and indeg[v] > 0:
            sinks.add(v)

    # Variable to store the total minimum number of strings (paths) needed
    total_paths = 0
    # Keep track of visited vertices during component finding
    visited = set()

    # Process vertices in a fixed order (sorted) to ensure deterministic behavior
    sorted_vertices = sorted(list(vertices)) 

    # Find weakly connected components using BFS
    for v_start in sorted_vertices:
        # If this vertex hasn't been visited yet, it starts a new component
        if v_start not in visited:
            
            # Set to store nodes in the current component
            component_nodes = set()
            # Queue for BFS
            q = deque([v_start])
            visited.add(v_start)
            
            # Use an index to track queue processing instead of popleft,
            # avoids issues if queue is modified during iteration. Although Python's deque popleft is O(1).
            head = 0
            while head < len(q):
                curr = q[head]
                head += 1
                component_nodes.add(curr) # Add the current node to the component set

                # Explore neighbors via outgoing edges
                # Check adj[curr] which gives list of nodes `neighbor` such that curr->neighbor edge exists
                for neighbor in adj.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
                
                # Explore neighbors via incoming edges
                # Check rev_adj[curr] which gives list of nodes `neighbor` such that neighbor->curr edge exists
                for neighbor in rev_adj.get(curr, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)

            # The logic assumes components found from `vertices` set always contain edges.
            # This is true because `vertices` only contains endpoints of edges S_i.

            # Identify sources and sinks within this component
            comp_sources = sources.intersection(component_nodes)
            comp_sinks = sinks.intersection(component_nodes)
            
            # Calculate the minimum paths needed for this component based on the hypothesis derived from samples:
            # The minimum number of paths for a component C is max(1, max(|Sources in C|, |Sinks in C|)).
            # A component always needs at least 1 path if it contains edges.
            # The number of paths required is driven by the maximum imbalance between pure sources and pure sinks.
            num_paths_comp = max(1, max(len(comp_sources), len(comp_sinks)))
            
            # Add the paths needed for this component to the total
            total_paths += num_paths_comp

    # Print the total minimum number of paths required
    print(total_paths)

# Execute the solve function
solve()