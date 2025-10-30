import sys

# Increase the recursion limit for deep DFS calls.
# N can be up to 2*10^5, so a limit slightly above this is necessary.
sys.setrecursionlimit(2 * 10**5 + 500) 

def solve():
    N = int(sys.stdin.readline())
    
    # Adjacency list to represent the tree: adj[u] stores tuples (v, weight)
    # Cities are 1-indexed, so we use N+1 size list and ignore index 0.
    adj = [[] for _ in range(N + 1)]
    total_sum_weights = 0

    # Read N-1 edges, build the adjacency list, and sum all edge weights.
    for _ in range(N - 1):
        A, B, C = map(int, sys.stdin.readline().split())
        adj[A].append((B, C))
        adj[B].append((A, C)) # Roads are bidirectional
        total_sum_weights += C

    # DFS function to find the farthest node and its distance from a starting point.
    # `max_dist_info` is a list [max_distance_found_so_far, node_at_that_distance].
    # It's passed by reference, so modifications inside the DFS are persistent.
    def dfs(u, parent, current_dist, max_dist_info):
        # If the current path to 'u' is longer than the max distance found so far, update.
        if current_dist > max_dist_info[0]:
            max_dist_info[0] = current_dist
            max_dist_info[1] = u
        
        # Explore neighbors of 'u'. Skip the parent to avoid going back immediately.
        for v, weight in adj[u]:
            if v != parent:
                dfs(v, u, current_dist + weight, max_dist_info)

    # Step 1: Find one endpoint of a diameter.
    # We start DFS from an arbitrary node (node 1 is convenient).
    # `max_dist_info_1` will store the farthest node from node 1 and its distance.
    max_dist_info_1 = [0, 1] # [max_distance, node_id]
    dfs(1, 0, 0, max_dist_info_1) # Parent of starting node can be 0 or any non-existent ID

    # The node farthest from node 1 is one endpoint of a diameter.
    farthest_node_from_1 = max_dist_info_1[1]

    # Step 2: Find the other endpoint of the diameter and its length.
    # Start a second DFS from the `farthest_node_from_1`.
    # The maximum distance found in this DFS will be the diameter's length.
    max_dist_info_2 = [0, farthest_node_from_1] # Reset max_dist_info for the new start
    dfs(farthest_node_from_1, 0, 0, max_dist_info_2)
    
    diameter_length = max_dist_info_2[0]

    # Calculate the minimum travel distance.
    # The formula is 2 * (total sum of all edge weights) - (length of the tree's diameter).
    # This comes from the fact that every edge must be traversed at least once.
    # Edges not on the diameter path are traversed twice (once to go into a branch, once to return).
    # Edges on the diameter path are traversed only once.
    result = 2 * total_sum_weights - diameter_length
    
    # Print the final result to standard output.
    sys.stdout.write(str(result) + '
')

# Call the solve function to execute the program logic.
solve()