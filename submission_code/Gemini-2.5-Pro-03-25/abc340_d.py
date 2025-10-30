# YOUR CODE HERE
import sys
import heapq

# Using sys.stdin.readline for potentially faster input reading compared to input()
# In competitive programming, reading large inputs can be a bottleneck,
# so this is a standard practice.
input = sys.stdin.readline 

def solve():
    """
    Solves the problem using Dijkstra's algorithm to find the minimum time 
    required to unlock stage N, starting from stage 1.

    The problem can be modeled as finding the shortest path in a directed graph
    where stages are nodes and actions represent edges with associated time costs.
    """
    
    # Read the number of stages N.
    N = int(input())
    
    # Store the parameters A_i, B_i, X_i for actions available at stages i=1 to N-1.
    # We use lists of size N-1. The index `k` in these lists corresponds to stage `k+1`.
    # This means that the parameters for actions at stage `u` (1-based index) 
    # are stored at index `u-1` in these lists.
    A = [0] * (N - 1) # Time costs for action 1 (go to stage i+1)
    B = [0] * (N - 1) # Time costs for action 2 (go to stage X_i)
    X = [0] * (N - 1) # Target stages for action 2 (1-based indices)
    
    for i in range(N - 1):
        # Read the parameters A_i, B_i, X_i for stage i+1.
        # The input format provides these values for i from 1 to N-1.
        # When reading into 0-indexed arrays, the i-th line corresponds to index i.
        line = list(map(int, input().split()))
        A[i] = line[0] 
        B[i] = line[1] 
        X[i] = line[2] 

    # Initialize the distance array `dist`. 
    # `dist[k]` will store the minimum time found so far to unlock stage k.
    # We use 1-based indexing for stages (nodes 1 to N), so the array size is N+1.
    # Initialize all distances to infinity, representing that stages are initially unreachable.
    dist = [float('inf')] * (N + 1)
    
    # Base case: Stage 1 is available from the start at time 0.
    dist[1] = 0 
    
    # Create a priority queue (min-heap) to manage stages to visit.
    # It stores tuples: (current_minimum_time, stage_index).
    # The heap property ensures we always process the stage reachable with the minimum time first.
    # Initialize with the starting stage 1 at time 0.
    pq = [(0, 1)] 
    
    # Main loop of Dijkstra's algorithm. Continues as long as there are stages to process.
    while pq:
        # Extract the stage `u` that currently has the minimum known time `d` from the priority queue.
        d, u = heapq.heappop(pq)
        
        # Check if this entry is outdated. If the extracted time `d` is greater than the 
        # already recorded minimum time `dist[u]`, it means we found a shorter path to `u` 
        # previously. So, we ignore this outdated entry and continue.
        if d > dist[u]:
            continue
            
        # Optimization: If the extracted stage `u` is the target stage N, we have found 
        # the minimum time required to reach it. Since Dijkstra explores paths in increasing 
        # order of length (time cost), the first time we extract N, its distance is finalized.
        # We can break the loop early.
        if u == N:
           break 

        # Actions are available only from stages 1 up to N-1. Stage N is the destination.
        if u < N:
            # Calculate the index `idx` in the A, B, X arrays corresponding to stage `u`.
            # Since stage `u` is 1-based, the corresponding index is `u-1`.
            idx = u - 1

            # Consider Action 1: Spend A[idx] seconds at stage `u` to unlock stage `u+1`.
            # The target stage is `v1 = u + 1`.
            # The time cost for this action is `cost1 = A[idx]`.
            v1 = u + 1
            cost1 = A[idx]
            
            # Relaxation step: Check if reaching `v1` through `u` yields a shorter path (less time).
            if dist[u] + cost1 < dist[v1]:
                # Update the minimum time to reach stage `v1`.
                dist[v1] = dist[u] + cost1
                # Add the updated information (new minimum time, stage) to the priority queue.
                # If `v1` was already in the queue with a higher time, this new entry will be processed first.
                heapq.heappush(pq, (dist[v1], v1))
            
            # Consider Action 2: Spend B[idx] seconds at stage `u` to unlock stage X[idx].
            # The target stage `v2` is given by `X[idx]` (which is already a 1-based index).
            # The time cost for this action is `cost2 = B[idx]`.
            v2 = X[idx] 
            cost2 = B[idx]
            
            # Relaxation step: Check if reaching `v2` through `u` yields a shorter path.
            if dist[u] + cost2 < dist[v2]:
                # Update the minimum time to reach stage `v2`.
                dist[v2] = dist[u] + cost2
                # Add the updated information to the priority queue.
                heapq.heappush(pq, (dist[v2], v2))

    # After the algorithm finishes, `dist[N]` contains the minimum time required to unlock stage N.
    # The problem statement implies that stage N is always reachable from stage 1.
    # Print the result.
    print(dist[N])

# Call the main solver function to execute the program logic.
solve()