# YOUR CODE HERE
import sys
import heapq

# Set higher recursion depth limit if needed, although Dijkstra is iterative.
# import sys 
# sys.setrecursionlimit(2000000) 

def solve():
    """
    Solves the problem using Dijkstra's algorithm on a graph constructed based on the operations.
    Nodes represent points 0 to N on a line. Edges represent the effect of operations.
    """
    
    # Read input values for N (sequence length) and M (number of operations)
    N, M = map(int, sys.stdin.readline().split())
    
    # Adjacency list representation of the graph using a dictionary.
    # Keys are source nodes, values are lists of dictionaries, each containing edge info.
    adj = {}

    # Helper function to add a directed edge to the graph.
    def add_edge(u, v, weight, op_idx, op_type):
        """
        Adds an edge from node u to node v with given weight and operation metadata.
        Initializes adjacency list for u if it doesn't exist.
        """
        if u not in adj: adj[u] = []
        adj[u].append({'target': v, 'weight': weight, 'op_idx': op_idx, 'op_type': op_type})

    # Process each of the M operations to build the graph edges.
    for i in range(M):
        L, R = map(int, sys.stdin.readline().split())
        
        # Add edge for Type 1 operation: Covers interval [L, R].
        # This is modeled as a directed edge from node L-1 to node R.
        # The weight/cost of using this operation is 1.
        # Metadata stored: operation index 'i' and type '1'.
        # Interpretation: If we have covered up to index L-1, we can cover up to R using this op.
        add_edge(L - 1, R, 1, i, 1)
        
        # Add edge for Type 2 operation: Covers complement interval(s) [1, L-1] and [R+1, N].
        # This is modeled as a directed edge from node R to node L-1.
        # The weight/cost is 1.
        # Metadata stored: operation index 'i' and type '2'.
        # Interpretation: This modeling choice aims to represent the effect of Type 2 operations
        # within the shortest path framework. Its correctness depends on whether this model accurately
        # captures the problem constraints and goal. It's one possible interpretation.
        add_edge(R, L - 1, 1, i, 2)

    # Initialize distances dictionary to store the shortest distance found so far to each node.
    # Use float('inf') to represent unreachable nodes initially.
    dist = {}
    # Parent dictionary to reconstruct the shortest path found by Dijkstra.
    # Stores tuple (previous_node, operation_index, operation_type) that led to the current node.
    parent = {}
    
    # Priority Queue for Dijkstra's algorithm. Stores tuples (current_cost, node).
    # Initialize with the starting node 0 at cost 0.
    pq = [(0, 0)] 
    dist[0] = 0
    # Set a sentinel parent for the starting node 0, indicating the path origin.
    parent[0] = (-1, -1, -1) # (prev_node = -1 means no parent, op_idx=-1, op_type=-1 mean no op)

    # Variable to store the minimum cost found to reach the target node N.
    min_cost = float('inf') 

    # Main loop of Dijkstra's algorithm. Continues as long as the priority queue is not empty.
    while pq:
        # Extract the node 'u' with the smallest distance 'd' from the priority queue.
        d, u = heapq.heappop(pq)

        # If the extracted distance 'd' is greater than the already known shortest distance to 'u',
        # this means we found a shorter path earlier, so skip processing this outdated element.
        if d > dist.get(u, float('inf')):
            continue

        # Check if the target node N has been reached. If yes, we have found the shortest path.
        # Record the cost and terminate the algorithm early.
        if u == N:
            min_cost = d
            break
        
        # If node 'u' has no outgoing edges defined in the adjacency list, skip to the next iteration.
        if u not in adj: continue

        # Iterate through all edges outgoing from the current node 'u'.
        for edge_info in adj[u]:
            v = edge_info['target']        # Target node of the edge
            weight = edge_info['weight']  # Cost of traversing this edge (always 1 in this problem)
            op_idx = edge_info['op_idx']  # Index of the operation (0 to M-1) corresponding to this edge
            op_type = edge_info['op_type'] # Type (1 or 2) of the operation for this edge

            # Calculate the potential new distance to node 'v' by going through 'u'.
            new_dist = d + weight
            
            # Relaxation step: Check if this path through 'u' offers a shorter distance to 'v'.
            if new_dist < dist.get(v, float('inf')):
               # Update the shortest distance found so far for node 'v'.
               dist[v] = new_dist
               # Record 'u' as the parent of 'v' on this new shortest path, storing the operation details.
               parent[v] = (u, op_idx, op_type)
               # Add node 'v' to the priority queue with its new distance. If 'v' was already in the queue,
               # this effectively updates its priority (heapq handles duplicates, effectively using the minimum).
               heapq.heappush(pq, (new_dist, v))

    # After Dijkstra's algorithm finishes:
    # Check if the target node N was reached (i.e., if N is present in the 'dist' dictionary).
    if N not in dist:
         # If N was not reached, it's impossible to cover all indices from 1 to N. Print -1.
         print("-1")
    else:
        # If N was reached, the minimum cost is dist[N]. Print this cost.
        min_cost = dist[N] 
        print(min_cost)
           
        # Initialize the list 'final_ops' to store the chosen operation type for each of the M operations.
        # Default type is 0 (do nothing).
        final_ops = [0] * M
           
        # Start backtracking from the target node N to reconstruct the sequence of operations used on the shortest path.
        curr = N
        while curr != 0:
            # Safety check: Ensure the current node exists in the parent dictionary. This should always be true if N is reachable.
            if curr not in parent:
                 # This state indicates an error potentially due to graph structure or algorithm flaw.
                 # Logging an error message might help debugging but is removed in final code.
                 # print(f"Error: path reconstruction failed at node {curr}.", file=sys.stderr)
                 break # Exit loop to prevent potential infinite loop or further errors.
                 
            # Retrieve the parent node 'prev' and the operation info (index 'op_idx', type 'op_type') that led to 'curr'.
            prev, op_idx, op_type = parent[curr]
            
            # Check if we have backtracked to the sentinel parent of node 0 (op_idx == -1). If so, path reconstruction is complete.
            if op_idx == -1:
                 break 

            # Record the type ('op_type') of the operation ('op_idx') used on this edge of the shortest path.
            # If an operation index appears multiple times (e.g., due to cycles, although Dijkstra avoids cycles on non-negative paths),
            # this assignment updates it; the last assignment during backtracking wins. Given path is simple, this is fine.
            final_ops[op_idx] = op_type
            
            # Move to the previous node ('prev') in the path to continue backtracking.
            curr = prev
            
            # Another safety check, primarily for robustness or complex graph scenarios.
            if curr == -1: # Explicit check for sentinel parent value
                 break
        
        # Print the final sequence of operation types (0, 1, or 2 for each operation), space-separated.
        print(*(final_ops))

# Execute the main function to solve the problem.
solve()