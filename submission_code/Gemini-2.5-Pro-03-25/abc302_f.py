# YOUR CODE HERE
import sys
from collections import deque

# Function to read input and solve the problem
def solve():
    # Read N (number of sets) and M (max integer value)
    # N: number of sets on the blackboard
    # M: maximum possible integer value in the sets
    N, M = map(int, sys.stdin.readline().split())

    # Create an adjacency list to represent the bipartite graph.
    # The graph has N set nodes and M integer nodes. Total nodes = N + M.
    # We use 0-based indexing for node representation for easier list indexing in Python.
    # Integer k (where 1 <= k <= M) maps to node index k-1.
    # Set i (where 1 <= i <= N) maps to node index M + i - 1.
    # This means integer nodes are indexed 0 to M-1, and set nodes are indexed M to M+N-1.
    adj = [[] for _ in range(N + M)]

    # Read the N sets from standard input and build the edges for the bipartite graph
    for i in range(N):
        # Read the size of the set A_i (number of elements in S_i)
        # This value is specified in the input format but not strictly needed for the algorithm logic here.
        _ = int(sys.stdin.readline()) 
        
        # Read the elements of the set S_i
        S_i = list(map(int, sys.stdin.readline().split()))
        
        # Determine the node index for the current set S_{i+1}.
        # In 0-based indexing for sets (0 to N-1), set i corresponds to node M+i.
        set_node_idx = M + i 
        
        # For each element in the set S_i
        for element in S_i:
            # Map the element value (1 to M) to its corresponding integer node index (0 to M-1).
            int_node_idx = element - 1 
            
            # Add bidirectional edges between the set node and its integer element nodes.
            # An edge (set_node_idx, int_node_idx) signifies that integer `element` is in set `i+1`.
            adj[set_node_idx].append(int_node_idx)
            adj[int_node_idx].append(set_node_idx)

    # Perform Breadth-First Search (BFS) to find the shortest path
    # from the node representing integer 1 to the node representing integer M.
    
    # Initialize a deque (double-ended queue) for the BFS process.
    q = deque()
    
    # Initialize an array to store distances from the start node. 
    # -1 indicates that a node has not been visited yet.
    dist = [-1] * (N + M)

    # Define the start and target nodes based on 0-based indexing
    start_node = 0       # Node index corresponding to integer 1
    target_node = M - 1  # Node index corresponding to integer M

    # Check if the start node (integer 1) exists in any set. 
    # If it doesn't, node 0 will have no neighbors in the graph. 
    # BFS will correctly handle this: it starts, finds no neighbors for node 0, 
    # the queue becomes empty, and the target will remain unreachable (dist = -1).
    # We don't need an explicit check, but it could potentially optimize slightly.

    # Initialize the BFS: distance to start node is 0, add start node to the queue.
    dist[start_node] = 0
    q.append(start_node)

    # BFS main loop: continues as long as there are nodes in the queue to process.
    while q:
        # Dequeue the current node `u` from the front of the queue.
        u = q.popleft()

        # If the current node `u` is the target node, we have found the shortest path.
        # Since BFS explores level by level, the first time we reach the target node,
        # it is via a shortest path. We can break the loop early.
        if u == target_node:
            break 

        # Explore neighbors `v` of the current node `u`.
        for v in adj[u]:
            # If neighbor `v` has not been visited yet (distance is -1)
            if dist[v] == -1:
                # Update the distance of `v` based on the distance to `u`.
                dist[v] = dist[u] + 1
                # Enqueue `v` to be processed later.
                q.append(v)

    # After BFS completes (either queue empty or target found), check the result.
    target_dist = dist[target_node]

    # Check if the target node was reached. If dist[target_node] is still -1, it means it's unreachable.
    if target_dist == -1:
        # If unreachable, it's impossible to obtain a set containing both 1 and M. Print -1.
        print("-1")
    else:
        # If the target node was reached, target_dist holds the length L of the shortest path.
        # The path in the bipartite graph alternates between integer nodes and set nodes.
        # A path between two integer nodes (like 1 and M) must have an even length L.
        # The path structure is: 1=k_0 - S_i1 - k_1 - S_i2 - ... - k_p - S_{i(p+1)} - M=k_{p+1}
        # The number of edges in this path is L = 2p + 2.
        # The number of set nodes involved is p + 1. Let this be k = p + 1.
        # Merging k sets requires k - 1 operations. So, minimum operations = (p+1) - 1 = p.
        # From L = 2p + 2, we can solve for p: p = (L - 2) / 2 = L/2 - 1.
        # Since L is even, integer division `//` gives the correct result.
        
        # Example: Path 1 - S_k - M. Length L=2. Minimum operations = 2//2 - 1 = 1 - 1 = 0. This is correct if a single set initially contains both 1 and M.
        # Example: Path 1 - S1 - k - S2 - M. Length L=4. Minimum operations = 4//2 - 1 = 2 - 1 = 1. This requires merging S1 and S2 (1 operation).
        
        # Print the minimum number of operations required.
        print(target_dist // 2 - 1)

# Call the main function to solve the problem using the defined logic.
solve()