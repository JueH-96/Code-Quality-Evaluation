import heapq
import sys

# Use a large value for infinity
INF = sys.maxsize

def solve():
    # Read graph structure
    N, M = map(int, sys.stdin.readline().split())
    bridges = []
    # Adjacency list: adj[u] is a list of (neighbor_v, bridge_index)
    # Using 1-based indexing for islands (1 to N)
    adj = [[] for _ in range(N + 1)]
    for i in range(M):
        u, v, t = map(int, sys.stdin.readline().split())
        bridges.append((u, v, t))
        adj[u].append((v, i)) # Store neighbor and bridge index (0-based)
        adj[v].append((u, i)) # Bidirectional

    Q = int(sys.stdin.readline())
    for _ in range(Q):
        # Read query specific information
        K = int(sys.stdin.readline())
        required_bridge_indices_1based = list(map(int, sys.stdin.readline().split()))
        # Convert 1-based bridge indices to 0-based
        required_bridge_indices_0based = [b - 1 for b in required_bridge_indices_1based]

        # Map original bridge index (0-based) to its index in the required list (0 to K-1)
        # This helps identify which bit in the mask corresponds to which required bridge
        req_index_map = {req_idx_0based: i for i, req_idx_0based in enumerate(required_bridge_indices_0based)}
        
        # Dijkstra on state space (mask, island)
        # dist[mask][island] stores the minimum time to reach 'island' having used
        # the set of required bridges indicated by 'mask'.
        # The mask is a bitmask of length K.
        # Use (1 << K) rows for masks, and N+1 columns for islands (1-based)
        dist = [[INF] * (N + 1) for _ in range(1 << K)]
        
        # Initial state: Start at island 1 with mask 0 (no required bridges used)
        dist[0][1] = 0
        # Priority queue stores tuples (time, mask, current_island)
        pq = [(0, 0, 1)] 

        # The target mask is when all K required bridges are used
        final_mask_val = (1 << K) - 1

        while pq:
            time, mask, u = heapq.heappop(pq)

            # If we found a shorter path to this state already, skip
            if time > dist[mask][u]:
                continue
            
            # Optimization: If we reached island N with the target mask, we can stop.
            # Dijkstra guarantees the first time (time, final_mask, N) is popped, 'time' is the minimum distance.
            if mask == final_mask_val and u == N:
                # Found the shortest path to the target state
                break

            # Explore neighbors using available bridges
            for v, bridge_idx in adj[u]:
                bridge_t = bridges[bridge_idx][2] # Get time cost

                if bridge_idx in req_index_map:
                    # This bridge is one of the K required bridges for this query
                    k_idx = req_index_map[bridge_idx]
                    new_mask = mask | (1 << k_idx) # Set the bit for this required bridge
                    
                    # Relax the edge in the state graph
                    if dist[mask][u] + bridge_t < dist[new_mask][v]:
                        dist[new_mask][v] = dist[mask][u] + bridge_t
                        heapq.heappush(pq, (dist[new_mask][v], new_mask, v))
                else:
                    # This bridge is not one of the required bridges for this query
                    new_mask = mask # Mask doesn't change

                    # Relax the edge in the state graph
                    if dist[mask][u] + bridge_t < dist[new_mask][v]:
                        dist[new_mask][v] = dist[mask][u] + bridge_t
                        heapq.heappush(pq, (dist[new_mask][v], new_mask, v))

        # The answer is the minimum time to reach island N with all K required bridges used
        # This is stored in dist[final_mask_val][N]. If it's still INF, it's unreachable,
        # but the problem guarantees connectivity.
        print(dist[final_mask_val][N])

solve()