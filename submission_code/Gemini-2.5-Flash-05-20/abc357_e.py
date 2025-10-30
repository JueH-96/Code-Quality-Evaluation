import sys

def solve():
    N = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    
    # Adjust 'a' to be 0-indexed internally.
    # If a_i points to vertex V (1-indexed), then adj[i-1] = V-1.
    adj = [x - 1 for x in a]

    # visited_state:
    # 0 = unvisited (not part of any traversal yet)
    # 1 = visiting (currently on the path of the active traversal)
    # 2 = visited (fully processed, its reachable_count is finalized)
    visited_state = [0] * N
    
    # reachable_count[u] stores the number of vertices reachable from vertex u.
    reachable_count = [0] * N
    
    # total_reachable_pairs accumulates the sum of reachable_count for all vertices.
    total_reachable_pairs = 0

    # Iterate through all vertices. If a vertex hasn't been fully processed (state 0),
    # start a traversal from it.
    for i in range(N):
        if visited_state[i] == 0: 
            
            path = [] # Stores nodes on the current path from the starting node 'i'
            curr = i
            
            # Traverse along the edges until we hit a node that has been visited before
            while visited_state[curr] == 0:
                visited_state[curr] = 1 # Mark 'curr' as 'visiting' (on current path)
                path.append(curr)
                curr = adj[curr]
            
            # Now, 'curr' is a node that has been visited before.
            # Its state can be 1 (part of the current path -> cycle detected)
            # or 2 (already fully processed by a previous traversal).
            
            if visited_state[curr] == 1: 
                # Case 1: 'curr' is currently on the path (state 1).
                # This means we found a cycle. 'curr' is the first node repeated in 'path'.
                
                # Find the starting index of the cycle within the 'path' list.
                cycle_start_index = path.index(curr)
                # The cycle nodes are the slice of 'path' from 'cycle_start_index' onwards.
                cycle_nodes = path[cycle_start_index:]
                cycle_len = len(cycle_nodes)
                
                # All nodes within this cycle can reach all other nodes in the same cycle.
                for node in cycle_nodes:
                    reachable_count[node] = cycle_len
                    visited_state[node] = 2 # Mark as fully processed
                    total_reachable_pairs += cycle_len
                    
                # Process nodes leading to the cycle (pre-cycle nodes).
                # These are nodes in 'path' before 'cycle_start_index'.
                
                # 'dist_to_cycle_start' tracks the number of steps from a pre-cycle node to reach 'curr' (the cycle entrance),
                # plus one for the node itself.
                # Example: If P -> Q -> R (where R is the cycle start), for P, dist is 2 (P, Q).
                # For Q, dist is 1 (Q).
                dist_to_cycle_start = 1 
                # Iterate backwards from the node just before 'curr' in 'path' to the start of 'path'.
                for j in range(cycle_start_index - 1, -1, -1):
                    node = path[j]
                    # The number of reachable nodes from 'node' is:
                    # (nodes on path from 'node' to 'curr') + (nodes in the cycle).
                    reachable_count[node] = dist_to_cycle_start + cycle_len
                    visited_state[node] = 2 # Mark as fully processed
                    total_reachable_pairs += reachable_count[node]
                    dist_to_cycle_start += 1
                        
            elif visited_state[curr] == 2: 
                # Case 2: 'curr' has been fully processed (state 2).
                # All nodes on the current 'path' lead to 'curr', which is already processed.
                # Their reachable count depends on 'reachable_count[curr]'.
                
                # 'dist_to_processed_node' tracks the number of steps from a node on 'path' to reach 'curr'.
                dist_to_processed_node = 1 
                # Iterate backwards through the 'path' list to compute reachable_count for each node.
                for j in range(len(path) - 1, -1, -1):
                    node = path[j]
                    # The number of reachable nodes from 'node' is:
                    # (nodes on path from 'node' to 'curr') + (nodes reachable from 'curr').
                    reachable_count[node] = dist_to_processed_node + reachable_count[curr]
                    visited_state[node] = 2 # Mark as fully processed
                    total_reachable_pairs += reachable_count[node]
                    dist_to_processed_node += 1
        
    print(total_reachable_pairs)

# Call the solve function to execute the program.
solve()