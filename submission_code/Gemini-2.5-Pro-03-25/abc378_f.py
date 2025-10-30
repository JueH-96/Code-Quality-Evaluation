# YOUR CODE HERE
import sys

# Set higher recursion depth if needed for potential deep DFS/BFS based approaches, 
# though current implementation is iterative. It's generally safe to increase it for competitive programming.
# sys.setrecursionlimit(200010) 

def solve():
    N = int(sys.stdin.readline())
    
    # Adjacency list representation of the tree
    # Using dictionary where keys are vertex IDs and values are lists of neighbors
    adj = {}

    def add_edge(u, v):
        # Use setdefault for cleaner addition of new vertices to the adjacency list
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    # Read N-1 edges and build adjacency list
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        add_edge(u, v)

    # Compute degrees of all vertices
    degrees = {}
    for i in range(1, N + 1):
       # Get the list of neighbors for vertex i, default to empty list if vertex has no edges (only possible for N=1)
       # The length of the list is the degree.
       degrees[i] = len(adj.get(i, []))

    # Identify vertices with degree 3 (S3)
    # We only need S3 vertices for our core logic.
    # Vertices with degree 2 (S2) are checked dynamically when computing neighbor counts.
    S3 = set()
    for i in range(1, N + 1):
        if degrees[i] == 3:
            S3.add(i)

    # Precompute counts of neighbors with degree 2 (cnt2) and degree 3 (cnt3) for each vertex w in S3
    # cnt2[w]: number of neighbors of w with degree 2
    # cnt3[w]: number of neighbors of w with degree 3 (this determines structure of T[S3])
    cnt2 = {}
    cnt3 = {}
    
    for w in S3:
        count2 = 0
        count3 = 0
        # Check neighbors for vertex w
        if w in adj: # Safety check, w must be in adj if N > 1
            for neighbor in adj[w]:
                # Check neighbor's degree using the precomputed degrees dictionary
                neighbor_deg = degrees.get(neighbor, 0) 
                if neighbor_deg == 2:
                    count2 += 1
                elif neighbor_deg == 3:
                    count3 += 1
        cnt2[w] = count2
        cnt3[w] = count3

    # Initialize total count of valid pairs {u, v}
    total_count = 0

    # Calculate C1: Contribution from paths of length 2 (form u - w - v)
    # For each vertex w in S3, we can form pairs {u, v} if w has at least two neighbors with degree 2.
    # The number of such pairs is "cnt2[w] choose 2".
    for w in S3:
        k2 = cnt2.get(w, 0) # Number of degree-2 neighbors for w
        if k2 >= 2:
            # Calculate combinations: k2 choose 2 = k2 * (k2 - 1) / 2
            total_count += k2 * (k2 - 1) // 2

    # Visited set to track vertices in S3 already processed as part of a path in T[S3]
    # This ensures each maximal path in T[S3] is processed exactly once.
    visited = set()

    # Calculate C2: Contribution from paths P' = w_1 - ... - w_k within T[S3] where k > 1
    # T[S3] is the subgraph induced by S3 vertices. Since T is a tree, T[S3] is a forest.
    # As max degree in T[S3] is 2, components are paths.
    # Iterate through potential start nodes of maximal paths in T[S3]
    for w_start in S3:
        if w_start in visited:
            continue

        # We initiate path traversal only from endpoints of maximal paths in T[S3].
        # Endpoints are characterized by having degree <= 1 within T[S3] (cnt3 <= 1).
        # Isolated vertices in T[S3] have cnt3 = 0.
        # Path endpoints connected to other S3 vertices have cnt3 = 1.
        if cnt3.get(w_start, 0) <= 1: 
            
            current_path_nodes = [] # Stores vertices w_i of the current path P'
            curr = w_start
            prev = -1 # Track previous node in traversal to avoid going back immediately

            # Iteratively trace the path composed of S3 vertices
            while curr != -1 :
                # Should always be true because we start from S3 and only step to S3 neighbors
                # Add current node to path and mark visited
                current_path_nodes.append(curr)
                visited.add(curr)
                
                # Find the next node in S3 along the path
                next_node = -1
                # A node can continue the path only if its degree in T[S3] is >= 1
                if cnt3.get(curr, 0) >= 1: 
                   if curr in adj:
                        for neighbor in adj[curr]:
                            # Check if neighbor is in S3 and not the node we just came from
                            if neighbor in S3 and neighbor != prev:
                                next_node = neighbor
                                break # Found the unique next node along the S3 path (degree in T[S3] <= 2)
                
                # If no valid next node exists (reached end of path in T[S3]), stop traversal
                if next_node == -1:
                    break 
                
                # Move to the next node
                prev = curr
                curr = next_node
            
            # Path P' = w_1, ..., w_m found (nodes stored in current_path_nodes).
            # Calculate contribution C2_P for this path
            m = len(current_path_nodes)
            if m > 1: # Only paths with k > 1 (m > 1 nodes) contribute to C2
                
                # Collect cnt2 values (number of degree-2 neighbors) for nodes on the path
                path_S = [cnt2.get(node, 0) for node in current_path_nodes]
                
                # Efficiently calculate Sum_{1 <= i < j <= m} S_i * S_j using the identity:
                # 2 * Sum_{i<j} S_i*S_j = (Sum S_i)^2 - Sum (S_i^2)
                TotalSum = sum(path_S)
                SumSq = sum(s*s for s in path_S)
                
                # Calculate the sum of products for pairs on this path
                C2_P = (TotalSum * TotalSum - SumSq) // 2
                total_count += C2_P
                
    # Print the final total count
    print(total_count)

# Execute the solve function when the script is run
solve()