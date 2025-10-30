# YOUR CODE HERE
import sys

# Increase recursion depth for deep trees
# Using a large value like 300k covers N=250k, but might hit system limits.
# 260k is hopefully a safe compromise. If this fails, iterative approaches
# for BFS/DFS might be necessary or a smaller limit could be tried.
try:
    # Set recursion depth high enough for deep trees up to N=250k
    sys.setrecursionlimit(260000) 
except OverflowError: 
     # Fallback if the requested limit is too high for the system
     # This part might need adjustment based on execution environment
     # For typical competitive programming platforms, this might not be an issue.
     pass


def solve():
    N = int(sys.stdin.readline())
    
    # The problem constraints state N >= 2, so N=0 case is not expected.
    # If N=0 was possible, we would handle it here (e.g., print nothing and return).

    # Adjacency list representation of the tree
    adj = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    # Bipartition coloring of the tree using BFS
    # color[i] = 0 for one partition (say Black), 1 for the other (say White)
    # -1 indicates uncolored/unvisited
    color = [-1] * (N + 1)
    
    # Since the graph is a tree (connected), we can start BFS from any node, e.g., vertex 1.
    q = [] 
    if N >= 1: # Check if graph has at least one vertex
        color[1] = 0 # Start coloring vertex 1 with color 0
        q.append(1)
        
        head = 0 # Use list as a queue with head index
        while head < len(q):
            u = q[head]
            head += 1
            
            # Explore neighbors of u
            for v in adj[u]:
                if color[v] == -1: # If neighbor v is uncolored
                    color[v] = 1 - color[u] # Assign opposite color
                    q.append(v) # Add to queue for exploration
                # Optional consistency check: In a tree (bipartite graph), 
                # an already colored neighbor must have the opposite color.
                # elif color[v] == color[u]: 
                #     # This should not happen for a tree. Handle error or ignore.
                #     pass 

    # Calculate initial degrees of all vertices
    degree = [0] * (N + 1)
    for i in range(1, N + 1):
         # Ensure index i is valid for adj list (size N+1 for vertices 1..N)
         if i < len(adj):
             degree[i] = len(adj[i])

    # Identify initial leaves (vertices with degree 1) and partition them by color
    leaves_B = [] # Store leaves with color 0
    leaves_W = [] # Store leaves with color 1
    
    # Keep track of removed vertices
    is_removed = [False] * (N + 1)

    for i in range(1, N + 1):
        if degree[i] == 1: # If vertex i is a leaf
            if color[i] == 0:
                leaves_B.append(i)
            else: # color[i] == 1
                leaves_W.append(i)

    # Store the pairs of vertices removed in each operation
    results = []

    # Perform N/2 operations as required
    for _ in range(N // 2):
        
        # Clean up the leaf lists: remove leaves that have already been marked as removed
        # This can happen if a node becomes a leaf and is added to the list,
        # but later removed before it's popped from the list. Check the list tail.
        while leaves_B and is_removed[leaves_B[-1]]:
             leaves_B.pop()
        while leaves_W and is_removed[leaves_W[-1]]:
             leaves_W.pop()

        # Problem guarantees that a valid move is always possible.
        # This implies that both leaves_B and leaves_W will be non-empty
        # when a move is needed.
        if not leaves_B or not leaves_W:
             # This block should theoretically not be reached.
             # If it is, there might be an issue with the problem statement understanding
             # or the implementation logic.
             break 

        # Select an arbitrary pair of leaves (u, v) with different colors.
        # Using list.pop() effectively treats the lists as stacks (LIFO).
        # Any valid pair is chosen; this corresponds to the simple O(N) strategy.
        u = leaves_B.pop()
        v = leaves_W.pop()

        # Record the pair removed in this step
        results.append((u, v))
        
        # Mark the chosen vertices as removed
        is_removed[u] = True
        is_removed[v] = True

        # Update the degrees of the neighbors of the removed vertices.
        # If a neighbor becomes a leaf (degree 1), add it to the appropriate leaf list.
        
        # Process neighbors of u
        # Check bounds to ensure u is a valid index for adj list
        if u < len(adj): 
            for neighbor_u in adj[u]:
                # Process neighbor only if it hasn't been removed already
                if not is_removed[neighbor_u]:
                    degree[neighbor_u] -= 1 # Decrease neighbor's degree
                    if degree[neighbor_u] == 1: # If neighbor becomes a leaf
                        # Add it to the correct leaf list based on its color
                        if color[neighbor_u] == 0:
                            leaves_B.append(neighbor_u)
                        else: # color is 1
                            leaves_W.append(neighbor_u)

        # Process neighbors of v
        # Check bounds to ensure v is a valid index for adj list
        if v < len(adj):
            for neighbor_v in adj[v]:
                # Process neighbor only if it hasn't been removed already
                if not is_removed[neighbor_v]:
                    degree[neighbor_v] -= 1 # Decrease neighbor's degree
                    if degree[neighbor_v] == 1: # If neighbor becomes a leaf
                        # Add it to the correct leaf list based on its color
                        if color[neighbor_v] == 0:
                            leaves_B.append(neighbor_v)
                        else: # color is 1
                            leaves_W.append(neighbor_v)

    # Prepare the output string. Printing all lines at once is faster.
    output_lines = []
    for pair in results:
        output_lines.append(f"{pair[0]} {pair[1]}")
    
    # Print the final result
    print("
".join(output_lines))

# Execute the main function to solve the problem
solve()