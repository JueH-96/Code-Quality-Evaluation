# YOUR CODE HERE
import heapq
import sys

def solve():
    # Read input: N vertices, M edges, K edges to set weight to 1
    N, M, K = map(int, sys.stdin.readline().split())
    
    # Represent the graph using an adjacency list. Using list of lists for potentially better performance.
    adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        # Add edge u -> v
        adj[u].append(v)
       
    # State Representation for DP/Shortest Path:
    # We want to find characteristics of paths based on the number of weight-1 edges used.
    # Let dist[k][v] be the minimum number of weight-0 edges required to reach vertex v
    # using exactly k weight-1 edges on the path from vertex 1.
    
    # Initialize distances to infinity. Size is (K+1) x (N+1)
    dist = [[float('inf')] * (N + 1) for _ in range(K + 1)]
    
    # Use Dijkstra's algorithm on the state space graph.
    # States are (vertex, k), where k is the number of weight-1 edges used.
    # The "distance" metric minimized by Dijkstra will be the number of weight-0 edges (z).
    # Priority Queue stores tuples: (current_dist_z, vertex, k)
    # Initialize priority queue with the start state: (0 weight-0 edges, vertex 1, 0 weight-1 edges)
    pq = [(0, 1, 0)] 
    dist[0][1] = 0 # Distance to start state (1, 0) is 0 weight-0 edges.

    while pq:
        # Extract state with minimum number of weight-0 edges (z)
        z, u, k = heapq.heappop(pq)

        # If we've already found a shorter path to this state (u, k), skip processing.
        # This check handles cycles and ensures efficiency.
        if z > dist[k][u]:
            continue

        # Explore neighbors of vertex u in the original graph
        for v in adj[u]:
            # Consider two possibilities for the edge (u, v):
            
            # Option 1: Treat edge (u, v) as having weight 0.
            # The number of weight-1 edges (k) remains the same.
            # The number of weight-0 edges (z) increases by 1.
            new_z_w0 = z + 1
            # If this path offers a smaller number of weight-0 edges to reach state (v, k), update distance.
            if new_z_w0 < dist[k][v]:
                dist[k][v] = new_z_w0
                # Push the updated state to the priority queue.
                heapq.heappush(pq, (new_z_w0, v, k))

            # Option 2: Treat edge (u, v) as having weight 1.
            # This is only possible if we haven't exceeded the total budget K of weight-1 edges.
            if k + 1 <= K:
                # The number of weight-1 edges increases to k+1.
                # The number of weight-0 edges (z) remains the same.
                new_z_w1 = z 
                # If this path offers a smaller number of weight-0 edges to reach state (v, k+1), update distance.
                if new_z_w1 < dist[k+1][v]:
                    dist[k+1][v] = new_z_w1
                    # Push the updated state to the priority queue.
                    heapq.heappush(pq, (new_z_w1, v, k+1))

    # After Dijkstra finishes, dist[k][N] contains Z(k), the minimum number of weight-0 edges 
    # required for a path from vertex 1 to vertex N using exactly k weight-1 edges.
    # Collect these minimum weight-0 edge counts for the target vertex N.
    min_z_at_N = [dist[k][N] for k in range(K+1)]

    # We want to find the maximum possible value X of the shortest distance from 1 to N.
    # The shortest distance is defined as the minimum number of weight-1 edges on any path from 1 to N.
    # We can binary search for this maximum value X.
    # The possible range for X is [0, K]. Shortest path cannot use more than K weight-1 edges.
    low = 0
    high = K + 1 # Use K+1 as exclusive upper bound for binary search range [0, K+1)
    ans = 0 # Stores the maximum X found to be achievable

    while low < high:
        # Calculate the midpoint X to check.
        mid_X = (low + high) // 2
        
        # Check function: Is it possible to choose K edges such that the shortest path distance is at least mid_X?
        # This check uses a condition derived from related minimum cost flow problems / Lagrangian relaxation techniques.
        # Condition: A shortest path distance of X is achievable if min_{0 <= k <= K} (Z(k) + k * X) <= K * X.
        
        min_val = float('inf')
        
        # Calculate min_{0 <= k <= K} (Z(k) + k * mid_X)
        for k in range(K + 1):
            # Only consider states k where vertex N is reachable (Z(k) is finite).
            if min_z_at_N[k] != float('inf'):
                 current_val = min_z_at_N[k] + k * mid_X
                 min_val = min(min_val, current_val)
        
        # Determine if mid_X is achievable based on the condition.
        possible = False
        # Handle the base case X=0 separately: A shortest path distance of 0 is always achievable
        # as long as N is reachable (which is guaranteed by the problem statement).
        if mid_X == 0:
             possible = True
        # If min_val is still infinity, it implies N is unreachable for any k <= K, 
        # which contradicts the problem statement. Check added for robustness.
        elif min_val != float('inf') and min_val <= K * mid_X:
             # Condition holds, mid_X is potentially achievable.
             possible = True
        # else: possible remains False, mid_X is not achievable.

        # Adjust binary search range based on the check result.
        if possible:
            # If mid_X is possible, it means we might be able to achieve an even higher distance.
            # Store mid_X as a potential answer and search in the upper half [mid_X + 1, high).
            ans = max(ans, mid_X)
            low = mid_X + 1
        else:
            # If mid_X is not possible, the maximum achievable distance must be less than mid_X.
            # Search in the lower half [low, mid_X).
            high = mid_X

    # Print the maximum possible shortest distance found.
    print(ans)

# Call the solve function to run the program.
solve()