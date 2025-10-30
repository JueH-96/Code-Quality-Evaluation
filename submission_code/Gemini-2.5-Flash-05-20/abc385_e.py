import sys
from collections import defaultdict

def solve():
    N = int(sys.stdin.readline())
    adj = defaultdict(list)
    degree = [0] * (N + 1) # Stores degrees of vertices in the original tree

    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
        degree[u] += 1
        degree[v] += 1

    # Initialize min_deletions with the maximum possible: deleting all N vertices.
    # The problem states "it is always possible to transform T into a Snowflake Tree",
    # so we will definitely find a valid snowflake tree.
    min_deletions = N 

    # Iterate through each vertex to consider it as the center 'c'
    for c in range(1, N + 1):
        # counts[k] stores how many neighbors 'b' of 'c' have 'k' potential leaves
        # (i.e., 'degree[b] - 1' neighbors other than 'c').
        counts = defaultdict(int)
        
        # max_k_val keeps track of the maximum 'k' encountered for degree[b]-1.
        # This determines the upper bound for 'y' in the next loop.
        max_k_val = -1 
        
        # Collect potential leaf counts for each neighbor of 'c'
        for neighbor_b in adj[c]:
            k = degree[neighbor_b] - 1 # Number of available children for 'b' if 'b' is a branch
            counts[k] += 1
            if k > max_k_val:
                max_k_val = k

        current_max_snowflake_size_for_c = 0
        
        # current_x_sum will store the sum of 'counts[j]' for 'j >= y_val'.
        # This effectively calculates 'x' (number of branches) for a given 'y_val'.
        # We iterate 'y_val' downwards from max_k_val to 0.
        # This allows us to efficiently update current_x_sum.
        
        current_x_sum = 0 
        
        # The loop for y_val needs to go from max_k_val down to 0 (inclusive).
        # If max_k_val is -1 (e.g., if 'c' has no neighbors, which is not possible for N>=3),
        # the loop range will be empty, and current_max_snowflake_size_for_c remains 0.
        for y_val in range(max_k_val, -1, -1):
            # Add counts for the current y_val.
            # This updates current_x_sum to be `sum(counts[j] for j >= y_val)`.
            current_x_sum += counts[y_val]
            
            # According to the definition, x (number of branches) must be at least 1.
            # y (number of leaves per branch) can be 0 based on sample outputs (e.g., star graph is snowflake).
            if current_x_sum >= 1: 
                # Calculate the size of the snowflake tree: 1 (center) + x (branches) + x*y (leaves)
                size = 1 + current_x_sum + (current_x_sum * y_val)
                current_max_snowflake_size_for_c = max(current_max_snowflake_size_for_c, size)
        
        # If a valid snowflake tree was found for this center 'c' (size > 0),
        # calculate deletions and update overall minimum.
        if current_max_snowflake_size_for_c > 0:
            deletions_for_c = N - current_max_snowflake_size_for_c
            min_deletions = min(min_deletions, deletions_for_c)
        
    # Print the final minimum number of deletions
    print(min_deletions)

# Call the solve function
solve()