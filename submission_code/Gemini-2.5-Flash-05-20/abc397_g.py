import heapq
import sys

# Constants for infinity
INF = float('inf')

def solve():
    N, M, K = map(int, sys.stdin.readline().split())
    edges_list = []
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        edges_list.append((u, v))

    # check(X_target):
    # This function determines if it's possible to choose exactly K edges to have weight 1
    # such that the shortest distance from vertex 1 to vertex N (where distance is sum of 0/1 weights)
    # is at least X_target.
    #
    # This is equivalent to finding the minimum number of 1-weight edges (`cost_needed`)
    # that *must* be selected to ensure that *any* path from 1 to N has at least X_target 1-weight edges.
    # If `cost_needed <= K`, then `X_target` is achievable.
    #
    # This `cost_needed` value can be found using a min-cost max-flow (MCMF) algorithm,
    # specifically, by computing the min-cost for 1 unit of flow in a constructed layered graph.
    # Since all edge costs are non-negative (0 or 1), Dijkstra's algorithm can be used for this.

    def check(X_target):
        # If X_target is 0, a shortest distance of 0 is always achievable.
        # This occurs if there is at least one path from 1 to N consisting entirely of 0-weight edges.
        # We can always choose our K edges such that they don't block all such paths.
        # Effectively, we can just not choose any edges on a 0-weight path.
        if X_target == 0:
            return True

        # Construct the layered graph for Dijkstra:
        # Nodes are (u, k_val), representing being at vertex `u` having accumulated `k_val` 1-weight edges.
        # `k_val` ranges from 0 to `X_target - 1`.
        # Add a source `s_idx` and a sink `t_idx` to this graph.
        
        # `get_layered_idx(u, k_val)` maps (original_vertex, 1s_count) to an integer index.
        # u is 1-indexed (from input). k_val is 0-indexed.
        # Index range for layered nodes: 0 to N * X_target - 1.
        # s_idx and t_idx are at the end.
        num_layered_nodes = N * X_target
        s_idx = num_layered_nodes
        t_idx = num_layered_nodes + 1
        total_graph_nodes = num_layered_nodes + 2

        # Adjacency list for Dijkstra graph: (neighbor_idx, edge_cost)
        adj = [[] for _ in range(total_graph_nodes)]
        
        def get_layered_idx(u, k_val):
            return (u - 1) * X_target + k_val

        # 1. Edge from global source `s_idx` to `(1, 0)`: capacity INF, cost 0.
        adj[s_idx].append((get_layered_idx(1, 0), 0))

        # 2. Edges from all `(N, k_val)` (where `k_val < X_target`) to global sink `t_idx`: capacity INF, cost 0.
        # These represent paths that reach N with fewer than X_target 1-weight edges.
        # If any such path is found by Dijkstra with 0 cost, it means no 1-weight edges are required.
        for k_val in range(X_target):
            n_layered_idx = get_layered_idx(N, k_val)
            adj[n_layered_idx].append((t_idx, 0))

        # 3. Edges based on original graph edges:
        for u_orig, v_orig in edges_list:
            for k_val in range(X_target): # Iterate through layers
                u_layer_idx = get_layered_idx(u_orig, k_val)

                # Option A: Use (u_orig, v_orig) as a 0-weight edge.
                # Path cost (number of 1-weights) does not increase.
                v_layer_idx_0cost = get_layered_idx(v_orig, k_val)
                adj[u_layer_idx].append((v_layer_idx_0cost, 0))

                # Option B: Use (u_orig, v_orig) as a 1-weight edge.
                # Path cost (number of 1-weights) increases by 1.
                # If `k_val + 1` reaches `X_target`, it means this path has accumulated `X_target` 1-weight edges.
                # If `v_orig` is `N`, this path satisfies the `X_target` condition, so it leads to the sink `t_idx` with cost 1.
                if k_val + 1 < X_target:
                    v_layer_idx_1cost = get_layered_idx(v_orig, k_val + 1)
                    adj[u_layer_idx].append((v_layer_idx_1cost, 1))
                else: # k_val + 1 == X_target
                    if v_orig == N:
                        adj[u_layer_idx].append((t_idx, 1))
        
        # Run Dijkstra from `s_idx` to `t_idx`
        min_costs = [INF] * total_graph_nodes
        min_costs[s_idx] = 0
        pq = [(0, s_idx)] # (current_cost, node_idx)

        while pq:
            current_cost, u_curr_idx = heapq.heappop(pq)

            if current_cost > min_costs[u_curr_idx]:
                continue

            for v_next_idx, edge_cost in adj[u_curr_idx]:
                if min_costs[u_curr_idx] + edge_cost < min_costs[v_next_idx]:
                    min_costs[v_next_idx] = min_costs[u_curr_idx] + edge_cost
                    heapq.heappush(pq, (min_costs[v_next_idx], v_next_idx))
        
        # `min_costs[t_idx]` is the minimum number of 1-weight edges required
        # to guarantee that *any* path from 1 to N has at least X_target 1-weight edges.
        # If this cost is less than or equal to K, then X_target is achievable.
        return min_costs[t_idx] <= K

    # Binary search for the maximum possible value of the shortest distance
    low = 0
    high = N # Maximum possible shortest distance is N-1 (or K, whichever is smaller)
    ans = 0

    while low <= high:
        mid = (low + high) // 2
        if check(mid):
            ans = mid
            low = mid + 1 # Try to achieve a larger shortest distance
        else:
            high = mid - 1 # mid is too high, try a smaller shortest distance
    
    sys.stdout.write(str(ans) + "
")

solve()