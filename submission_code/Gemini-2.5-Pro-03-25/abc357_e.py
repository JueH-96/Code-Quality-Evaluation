# YOUR CODE HERE
import sys

# Increase recursion depth limit for deep graphs or large components
try:
    # Set a large enough limit, N + some buffer
    # For N = 2*10^5, needs more than default 1000
    # Added some buffer space beyond N
    sys.setrecursionlimit(2 * 10**5 + 50) 
except (OverflowError, ValueError): 
    # Some systems might restrict changing recursion depth, ignore error if occurs
    # ValueError can occur on some systems too e.g. Windows
    pass 

def solve():
    N = int(sys.stdin.readline())
    if N == 0: # Handle empty graph case although constraints state N>=1
         print(0)
         return

    a = list(map(int, sys.stdin.readline().split()))
    # Adjust 'a' to be 0-indexed node successors
    # Input gives 1-based vertex numbers. Edge i -> a_i corresponds to 
    # 0-based edge from vertex i-1 to vertex a[i-1]-1
    succ = [x - 1 for x in a] # Successor array, using 0-based indices

    # Data structures using N size for 0..N-1 indices
    visited = [0] * N # 0: unvisited, 1: visiting, 2: visited (fully processed)
    component_id = [-1] * N # Stores the component ID for each node
    is_cycle_node = [False] * N # Boolean flag, True if node is part of a cycle
    cycle_len = {} # Dictionary mapping component ID to the length of its cycle
    # Dictionary storing the list of cycle nodes for each component ID
    component_cycles = {} 
    comp_id_counter = 0 # Counter for assigning unique component IDs, starting from 1

    # Build the reversed graph adjacency list adj_rev
    # adj_rev[v] contains list of nodes u such that there is an edge u -> v in the original graph
    adj_rev = [[] for _ in range(N)]
    for i in range(N):
        # Check if the successor index is valid (0 <= succ[i] < N).
        # This is guaranteed by problem constraints 1 <= a_i <= N.
        if 0 <= succ[i] < N:
             adj_rev[succ[i]].append(i)

    # Step 1 & 2: Find cycles using DFS and assign component IDs
    # This phase identifies the structure of the graph: components, cycles, and trees feeding into cycles.
    for i in range(N):
        if visited[i] == 0:
            # Start path tracing from an unvisited node i
            path = [] # Keep track of nodes in the current path
            curr = i
            
            # Follow the unique outgoing edge from each node until a visited node is encountered
            while visited[curr] == 0:
                visited[curr] = 1 # Mark node as 'visiting' (on current exploration path)
                path.append(curr)
                curr = succ[curr] # Move to the next node

            # Check the state of the node where path tracing stopped
            if visited[curr] == 1: # Found a cycle: revisited a node marked 'visiting'
                
                # Verify curr is indeed on the current path. This must be true for functional graphs.
                try:
                     # Find the index where the cycle begins in the path list
                     cycle_start_index = path.index(curr)
                except ValueError:
                     # This block should ideally not be reached under problem constraints.
                     # If curr marked 'visiting' is not in path, it implies an issue.
                     # Mark nodes as visited=2 to prevent re-processing this path and continue.
                     for node in path:
                         visited[node] = 2 
                     continue 

                # Successfully identified a cycle and its start index
                comp_id_counter += 1
                                
                # Extract the nodes forming the cycle
                current_cycle = path[cycle_start_index:]
                k_len = len(current_cycle) # Cycle length
                
                # Store cycle information mapped to the new component ID
                component_cycles[comp_id_counter] = current_cycle 
                cycle_len[comp_id_counter] = k_len

                # Assign component ID and mark nodes in the 'tail' (path leading into the cycle)
                for node_idx in range(cycle_start_index):
                    node = path[node_idx]
                    visited[node] = 2 # Mark as fully processed
                    component_id[node] = comp_id_counter

                # Assign component ID and mark cycle nodes
                for node in current_cycle:
                    visited[node] = 2 # Mark as fully processed
                    component_id[node] = comp_id_counter
                    is_cycle_node[node] = True # Mark as a cycle node

            elif visited[curr] == 2: # Merged into an already processed component
                # The path starting from i eventually leads to node curr which is part of an existing component.
                target_comp_id = component_id[curr] # Get the component ID of curr
                
                # Assign the same component ID to all nodes in the current path
                if target_comp_id != -1: # Check if component ID is valid
                    for node in path:
                        visited[node] = 2 # Mark node as fully processed
                        component_id[node] = target_comp_id
                # else: This indicates an error state - visited[curr]=2 but no component_id assigned.
                
    
    # Step 4-6: Compute the total number of reachable pairs using properties derived.
    # The total count is Sum_{u} |R(u)| = Sum_{components C} ( Sum_{c in cycle} S(c) + k * N_C )
    # where S(c) is sum of distances from cycle node c to nodes in its tree in reversed graph G'
    # k is cycle length, N_C is total nodes in component C.
    
    total_pairs = 0
    
    # Memoization dictionary for the DFS compute phase to store results for (size, S) per node
    memo_dfs = {}

    # Recursive function to compute subtree size and sum of distances S(u) in reversed graph G'
    # S(u) = Sum_{v child of u in G'} (S(v) + size(v))
    def dfs_compute_memo(u):
        # Return cached result if available
        if u in memo_dfs:
            return memo_dfs[u]

        current_size = 1 # Initialize size for node u itself
        current_S = 0 # Initialize sum of distances for node u's subtree rooted at u
        
        # Explore neighbors (children) in the reversed graph G'
        for v in adj_rev[u]:
            # Process neighbor v only if it belongs to the same component as u
            # AND v is NOT a cycle node itself (to stay within the tree part attached to u)
            if component_id[v] != -1 and component_id[v] == component_id[u] and not is_cycle_node[v]:
                 # Recursive call for child v
                 subtree_size, subtree_S = dfs_compute_memo(v)
                 # Aggregate results using the Dynamic Programming relation:
                 current_size += subtree_size # Add size of child's subtree
                 current_S += subtree_S + subtree_size # Add child's S value plus size (because path from u to nodes in v's subtree is 1 edge longer)

        # Store result in cache before returning
        memo_dfs[u] = (current_size, current_S)
        return current_size, current_S

    # Iterate through all identified components
    # Use comp_id_counter which tracks the number of components found
    for comp_id in range(1, comp_id_counter + 1):
           # Retrieve component information using its ID
           k = cycle_len[comp_id] # Cycle length
           found_cycle = component_cycles[comp_id] # List of nodes in the cycle
           
           current_comp_dist_sum = 0 # Accumulator for Sum S(c) over cycle nodes c
           current_comp_node_count = 0 # Accumulator for total nodes N_C in the component
                       
           # Compute size and S value for each tree rooted at a cycle node c
           for node in found_cycle:
               # Initiate DFS computation from cycle node 'node'.
               # Memoization handles overlapping computations if any (not expected here) and avoids recomputing.
               tree_size, tree_S = dfs_compute_memo(node)
               # Accumulate the Sum of distances S(c) for all cycle nodes c
               current_comp_dist_sum += tree_S
               # Accumulate the total size of trees rooted at cycle nodes c (this gives total nodes N_C in component)
               current_comp_node_count += tree_size
           
           # Add this component's contribution to the total count of reachable pairs
           # Contribution formula: Sum_{c in Cycle} S(c) + k * (Sum_{c in Cycle} size(c))
           # This simplifies to: current_comp_dist_sum + k * current_comp_node_count
           total_pairs += current_comp_dist_sum + k * current_comp_node_count

    # Print the final total count
    print(total_pairs)

# Execute the main function to solve the problem
solve()