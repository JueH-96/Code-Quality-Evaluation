import heapq
import sys

def solve():
    N = int(sys.stdin.readline())

    # A_costs[i], B_costs[i], X_stages[i] store data for actions AT stage i.
    # These are defined for i from 1 to N-1.
    # Using 1-based indexing for stages, so arrays are size N.
    # Index 0 is unused. A_costs[N-1], B_costs[N-1], X_stages[N-1] are the last used.
    A_costs = [0] * N 
    B_costs = [0] * N
    X_stages = [0] * N # X_stages[i] is the target stage for B_i action

    for i in range(1, N): # Loop for i = 1, ..., N-1
        # Read A_i, B_i, X_i for stage i
        parts = sys.stdin.readline().split()
        A_costs[i] = int(parts[0])
        B_costs[i] = int(parts[1])
        X_stages[i] = int(parts[2])

    # dist[j] stores the minimum time to make stage j playable.
    # Using 1-based indexing for stages, so array size N+1. Index 0 unused.
    dist = [float('inf')] * (N + 1)
    dist[1] = 0 # Cost to make stage 1 playable is 0.

    # Priority queue stores (cost, stage_id) tuples.
    # heapq module implements a min-priority queue.
    pq = [(0, 1)] # Start with stage 1 at cost 0.

    while pq:
        current_cost, u = heapq.heappop(pq)

        # If this path is already known to be suboptimal, skip.
        if current_cost > dist[u]:
            continue
        
        # If we extract stage N from PQ, its shortest path is found.
        # Dijkstra guarantees this because edge weights (times) are non-negative.
        if u == N:
            break 

        # Actions from stage u (1 <= u <= N-1):
        # Note: A_costs[u], B_costs[u], X_stages[u] are defined because u < N
        # (due to the `if u == N: break` above) and actions are for stages 1 to N-1.
        # The input loop `for i in range(1,N)` ensures data exists for these indices.

        # Action A: Use A_costs[u] to unlock stage u+1.
        # This path costs current_cost + A_costs[u] to make u+1 playable.
        cost_to_neighbor_a = current_cost + A_costs[u]
        neighbor_a = u + 1 # Target stage for Action A is always u+1.
        
        if cost_to_neighbor_a < dist[neighbor_a]:
            dist[neighbor_a] = cost_to_neighbor_a
            heapq.heappush(pq, (dist[neighbor_a], neighbor_a))

        # Action B: Use B_costs[u] to unlock stage X_stages[u].
        # This path costs current_cost + B_costs[u] to make X_stages[u] playable.
        cost_to_neighbor_b = current_cost + B_costs[u]
        neighbor_b = X_stages[u] # Target stage for Action B is X_u.
        
        if cost_to_neighbor_b < dist[neighbor_b]:
            dist[neighbor_b] = cost_to_neighbor_b
            heapq.heappush(pq, (dist[neighbor_b], neighbor_b))
                
    # The minimum time to make stage N playable.
    sys.stdout.write(str(dist[N]) + "
")

# Call the solver function.
solve()