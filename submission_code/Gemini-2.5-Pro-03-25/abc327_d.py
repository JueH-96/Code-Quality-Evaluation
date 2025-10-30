# YOUR CODE HERE
import sys
# Increase recursion depth for potential DFS alternative, though BFS is used here
# sys.setrecursionlimit(200005) 
from collections import deque

def solve():
    # Read N (number of elements in sequence X) and M (length of sequences S, T)
    N, M = map(int, sys.stdin.readline().split())
    
    # Read sequence A = (A_1, ..., A_M)
    A = list(map(int, sys.stdin.readline().split()))
    # Read sequence B = (B_1, ..., B_M)
    B = list(map(int, sys.stdin.readline().split()))

    # The problem asks if there exists a binary sequence X = (X_1, ..., X_N)
    # such that X_{A_i} != X_{B_i} for all i=1..M.
    # This is equivalent to asking if the graph G with vertices {1, ..., N}
    # and edges {(A_i, B_i) for i=1..M} is bipartite (2-colorable).
    # We can check bipartiteness using Breadth-First Search (BFS).

    # Build adjacency list representation of the graph G.
    # Vertices are numbered 1 to N.
    # We use a list of lists `adj` of size N+1, index 0 is unused.
    adj = [[] for _ in range(N + 1)]
    for i in range(M):
        u, v = A[i], B[i]
        # Add an undirected edge between u and v.
        # A self-loop (u=v) corresponds to the condition X_u != X_u, which is impossible.
        # The BFS algorithm naturally detects this as an odd cycle of length 1.
        adj[u].append(v)
        adj[v].append(u)

    # Initialize color array to store the color assigned to each vertex during BFS.
    # -1: uncolored/unvisited
    # 0: color 0
    # 1: color 1
    color = [-1] * (N + 1) 
    
    # Flag to track if the entire graph is bipartite. Assume true initially.
    is_bipartite = True

    # Use a deque for efficient queue operations in BFS.
    q = deque() 

    # Iterate through all vertices from 1 to N.
    # This ensures that all connected components of the graph are processed.
    # If the graph is disconnected, BFS needs to be started from a vertex in each component.
    for i in range(1, N + 1):
        # If vertex i has not been colored yet, it means we have found a starting vertex
        # of a new connected component that needs to be checked for bipartiteness.
        if color[i] == -1: 
            
            # Start BFS for this component.
            # Assign the starting vertex color 0. Could be 1 as well, choice is arbitrary.
            color[i] = 0 
            q.append(i)  # Add the starting vertex to the queue.
            
            # Flag to track if the current component is bipartite. Assume true.
            component_is_bipartite = True
            
            # Standard BFS loop: continues as long as there are vertices in the queue.
            while q:
                u = q.popleft() # Dequeue a vertex u.
                
                # Explore neighbors of u.
                for v in adj[u]:
                    # Case 1: Neighbor v is uncolored.
                    if color[v] == -1: 
                        # Assign the opposite color of u to v.
                        color[v] = 1 - color[u] 
                        q.append(v) # Enqueue v for later processing.
                    # Case 2: Neighbor v is already colored.
                    # Check if v has the same color as u.
                    elif color[v] == color[u]: 
                        # If they have the same color, it implies an edge connects two vertices
                        # within the same partition set, which means an odd cycle exists.
                        # The graph (and this component) is not bipartite.
                        component_is_bipartite = False
                        # Optimization: Clear the queue to stop BFS for this component quickly.
                        q.clear() 
                        break # Exit the loop over neighbors of u. We found a conflict.
                
                # If the inner loop (over neighbors) was broken due to a conflict,
                # we should also break the outer while loop (BFS loop for the component).
                if not component_is_bipartite:
                     break 
            
            # If a conflict was found within this component (it's not bipartite),
            # then the entire graph cannot be bipartite.
            if not component_is_bipartite:
                is_bipartite = False # Mark the whole graph as not bipartite.
                break # Exit the main loop over vertices; no need to check further components.

    # After checking all components, print the final result based on the is_bipartite flag.
    if is_bipartite:
        print("Yes") # The graph is bipartite, meaning a valid sequence X exists.
    else:
        print("No") # The graph is not bipartite, meaning no such sequence X exists.

# Execute the main function to solve the problem
solve()