import heapq
import sys

def solve():
    N, M, X = map(int, sys.stdin.readline().split())

    # adj[u] stores v if u -> v is an edge in the original graph
    adj = [[] for _ in range(N + 1)]
    # rev_adj[u] stores v if v -> u is an edge in the original graph
    # (i.e., u -> v is an edge in the reversed graph, if we are at u)
    rev_adj = [[] for _ in range(N + 1)] 

    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        rev_adj[v].append(u) # If u->v original, then v is a predecessor of u in reversed graph.
                             # If current node is X, and Y->X was original edge, then X->Y is reversed edge.
                             # So rev_adj[target_node] stores source_nodes.
                             # When at node 'u' in reversed graph, want to move u -> v_neighbor.
                             # This means v_neighbor -> u was an original edge.
                             # So v_neighbor is in rev_adj[u].

    # dist[vertex_idx][direction_state]
    # direction_state: 0 for original graph, 1 for reversed graph
    # Initialize distances to infinity
    dist = [[float('inf')] * 2 for _ in range(N + 1)]

    # Priority queue stores tuples: (cost, vertex_idx, direction_state)
    pq = []

    # Initial state: at vertex 1, edges are in original direction, cost is 0
    dist[1][0] = 0
    heapq.heappush(pq, (0, 1, 0)) 

    while pq:
        current_cost, u, dir_state = heapq.heappop(pq)

        # If we found a shorter path to this state already, skip
        if current_cost > dist[u][dir_state]:
            continue

        # Option 1: Move along a directed edge (cost = 1)
        if dir_state == 0:  # Edges are in original direction, use adj[u]
            for v_neighbor in adj[u]:
                new_cost = current_cost + 1
                if new_cost < dist[v_neighbor][0]:
                    dist[v_neighbor][0] = new_cost
                    heapq.heappush(pq, (new_cost, v_neighbor, 0))
        else:  # dir_state == 1, Edges are reversed
               # An original edge v_orig -> u becomes a reversed edge u -> v_orig.
               # We are at u. We want to move to v_orig.
               # Such v_orig are listed in rev_adj[u].
            for v_neighbor in rev_adj[u]: 
                new_cost = current_cost + 1
                if new_cost < dist[v_neighbor][1]:
                    dist[v_neighbor][1] = new_cost
                    heapq.heappush(pq, (new_cost, v_neighbor, 1))
        
        # Option 2: Reverse all edges (cost = X)
        # Current state is (u, dir_state) with cost current_cost.
        # New state after reversing is (u, 1 - dir_state).
        new_dir_state = 1 - dir_state
        cost_after_reversal = current_cost + X
        
        if cost_after_reversal < dist[u][new_dir_state]:
            dist[u][new_dir_state] = cost_after_reversal
            heapq.heappush(pq, (cost_after_reversal, u, new_dir_state))
                
    result = min(dist[N][0], dist[N][1])
    
    # The problem guarantees N is reachable, so result will not be float('inf').
    # Costs are integers, print as integer.
    print(int(result))

# Call the solver function
solve()