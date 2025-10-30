import sys

# Setting a higher recursion depth limit is a common practice in competitive programming for Python,
# although this solution is iterative and doesn't rely on deep recursion.
# sys.setrecursionlimit(4 * 10**5) 

def solve():
    # Read the number of vertices
    N = int(sys.stdin.readline())
    
    # Adjacency list representation of the tree. Vertices are 0-indexed.
    adj = [[] for _ in range(N)]
    
    # Read N-1 edges and build the adjacency list
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        u -= 1 # Adjust to 0-based index
        v -= 1 # Adjust to 0-based index
        adj[u].append(v)
        adj[v].append(u)

    # Compute the degree of each vertex
    degrees = [len(adj[i]) for i in range(N)]

    # Precompute L[v] for each vertex v:
    # L[v] = count of neighbors of v that are leaves (degree 1) in the original tree T.
    L = [0] * N
    for i in range(N):
        # Iterate through neighbors of vertex i
        for neighbor in adj[i]:
            # Check if the neighbor is a leaf node
            if degrees[neighbor] == 1:
                L[i] += 1

    max_K = 0 # Initialize the maximum number of vertices found in any Snowflake Tree subgraph.

    # Iterate through each vertex C, considering it as the potential center of a Snowflake Tree.
    for C in range(N):
        
        # If a vertex has degree 0, it cannot be a center. 
        # This check is technically redundant for a tree with N>=3, as all vertices have degree >= 1.
        if degrees[C] == 0: continue 

        # List to store k_v values for neighbors v of the current potential center C.
        # k_v = number of leaf neighbors of v, excluding C itself if C is a leaf.
        neighbor_leaf_counts = [] 
        
        # Iterate through each neighbor v of the potential center C
        for v in adj[C]:
            # Calculate k_v using the precomputed L[v].
            kv = L[v]
            
            # If the potential center C itself is a leaf node (degree 1),
            # then C must have been counted in L[v] (since deg(C)=1).
            # We need the count of leaf neighbors of v *excluding* C, so we subtract 1.
            if degrees[C] == 1:
                 kv -= 1
            
            # Append the calculated k_v value to the list for this center C.
            neighbor_leaf_counts.append(kv)

        # Sort the k_v values in descending order. This facilitates finding the optimal 'y'
        # for each possible number of branches 'x'.
        neighbor_leaf_counts.sort(reverse=True)
        
        # Iterate through possible values of 'x' (number of branches emanating from the center C).
        # 'x' ranges from 1 up to the degree of C.
        num_neighbors = len(neighbor_leaf_counts)
        for x in range(1, num_neighbors + 1):
            # To form a Snowflake Tree with 'x' branches, each corresponding B_i vertex must support
            # at least 'y' leaves. With the k_v values sorted descendingly, the x-th value
            # (k'_x = neighbor_leaf_counts[x-1]) is the minimum k_v among the top x neighbors.
            # This minimum k_v determines the maximum possible value for 'y'. We choose y = k'_x.
            y = neighbor_leaf_counts[x-1]
            
            # A valid Snowflake Tree requires positive integers x and y (x >= 1, y >= 1).
            # The loop ensures x >= 1. We must check if y >= 1.
            if y >= 1:
                # Calculate the total number of vertices in the Snowflake Tree formed with
                # parameters x and y. The formula is 1 (center C) + x (intermediate B_i vertices) + x*y (leaf L_{i,j} vertices).
                V = 1 + x * (1 + y) 
                
                # Update the overall maximum Snowflake Tree size found so far across all potential centers and configurations.
                if V > max_K:
                     max_K = V
            else:
                 # If y < 1 (i.e., y=0 for the x-th neighbor count), then any subsequent neighbor (x+1, x+2, ...)
                 # will also have k_v <= y < 1, because the counts are sorted descendingly.
                 # Thus, no valid Snowflake Tree (with y >= 1) can be formed using more than x branches.
                 # We can break the inner loop early for this center C.
                 break 
        
    # The problem statement guarantees that it's always possible to transform the given tree T
    # into a Snowflake Tree. The smallest Snowflake Tree has 3 vertices.
    # Therefore, for N >= 3, the calculated max_K will be at least 3.
    
    # The minimum number of vertices that must be deleted is the difference between the
    # total number of vertices N and the size of the largest Snowflake Tree subgraph found (max_K).
    print(N - max_K)

# Execute the solver function
solve()