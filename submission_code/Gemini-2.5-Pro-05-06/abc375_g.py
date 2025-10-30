import heapq
import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    
    edges_input = [] # To store original edges for the final check
    adj = [[] for _ in range(N + 1)]

    for i in range(M):
        u, v, c = map(int, sys.stdin.readline().split())
        edges_input.append((u, v, c))
        adj[u].append((v, c))
        adj[v].append((u, c))
        
    infinity = float('inf')

    def dijkstra(start_node, current_adj):
        distances = [infinity] * (N + 1)
        distances[start_node] = 0
        pq = [(0, start_node)] # (distance, node_id)
        
        while pq:
            dist, u = heapq.heappop(pq)
            
            if dist > distances[u]: # Already found a shorter path to u
                continue
            
            for v, weight in current_adj[u]:
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    heapq.heappush(pq, (distances[v], v))
        return distances

    dist1 = dijkstra(1, adj) # Shortest distances from node 1
    distN = dijkstra(N, adj) # Shortest distances from node N (to all other nodes)

    S_all = dist1[N] # Shortest path from 1 to N using all roads

    # Calculate P1_prime (number of shortest paths from 1 to u, capped at 2)
    # Sort nodes by distance from 1, to process in topological order of the shortest path DAG
    nodes_by_dist1 = sorted(range(1, N + 1), key=lambda i: dist1[i])
    
    P1_prime = [0] * (N + 1)
    if dist1[1] == 0: # Node 1 is the start
         P1_prime[1] = 1

    for u_node in nodes_by_dist1:
        if u_node == 1: # Base case for node 1 already handled
            continue
        if dist1[u_node] == infinity: # Node u_node is unreachable from 1
            continue
        
        current_val_for_P1_u_node = 0
        # For P1_prime[u_node], sum P1_prime[v_prev] for all incoming edges (v_prev, u_node)
        # that are part of a shortest path to u_node.
        # adj[u_node] gives (neighbor, weight). If neighbor is v_prev, this is an edge (v_prev, u_node) or (u_node, v_prev).
        # We are interested in (v_prev, u_node) such that d1[v_prev] + weight == d1[u_node].
        for v_prev, weight in adj[u_node]: 
            if dist1[v_prev] + weight == dist1[u_node]:
                current_val_for_P1_u_node += P1_prime[v_prev]
                if current_val_for_P1_u_node >= 2: # Cap sum at 2
                    current_val_for_P1_u_node = 2
                    # Optimization: if sum is already 2 or more, further additions won't change it.
                    # However, we must iterate all v_prev to ensure correctness if P1_prime[v_prev] could be 0.
                    # The min(2, sum + P1_prime[v_prev]) structure is safer.
        # Re-writing sum carefully:
    # P1_prime[u_node] was initialized to 0.
    # The loop for u_node starts from the second element of nodes_by_dist1 if node 1 is first.
    # This ensures P1_prime[v_prev] is computed before P1_prime[u_node] if d1[v_prev] < d1[u_node].
    # If d1[v_prev] == d1[u_node] (only if weight is 0, not possible here as C_i >= 1), order matters.
    # With C_i >= 1, d1[v_prev] < d1[u_node] if dist1[v_prev] + weight == dist1[u_node].

    # Corrected P1_prime loop:
    for u_node_idx in range(len(nodes_by_dist1)):
        u_node = nodes_by_dist1[u_node_idx]
        if u_node == 1: continue 
        if dist1[u_node] == infinity: continue
        
        val = 0
        for v_prev, weight in adj[u_node]:
            if dist1[v_prev] + weight == dist1[u_node]:
                val = min(2, val + P1_prime[v_prev])
        P1_prime[u_node] = val
        
    # Calculate PN_prime (number of shortest paths from u to N, capped at 2)
    # Iterate nodes by decreasing d1 order.
    nodes_by_dist1_rev = nodes_by_dist1[::-1] 

    PN_prime = [0] * (N + 1)
    if distN[N] == 0: # Node N is the destination for these paths
        PN_prime[N] = 1
        
    for u_node_idx in range(len(nodes_by_dist1_rev)):
        u_node = nodes_by_dist1_rev[u_node_idx]
        if u_node == N: continue
        if distN[u_node] == infinity: # if u_node cannot reach N
            continue
            
        val = 0
        # For PN_prime[u_node], sum PN_prime[v_next] for all outgoing edges (u_node, v_next)
        # that are part of a shortest path from u_node to N.
        # Such an edge satisfies distN[v_next] + weight == distN[u_node].
        for v_next, weight in adj[u_node]:
            if distN[v_next] + weight == distN[u_node]:
                val = min(2, val + PN_prime[v_next])
        PN_prime[u_node] = val
    
    total_shortest_paths_count_capped = P1_prime[N]
    
    results = []
    for i in range(M):
        u, v, c = edges_input[i]
        
        ans_for_this_edge = "No" 
        
        # Check direction u -> v
        # Condition: d1[u] + c == d1[v] (edge is tight for d1)
        # AND d1[u] + c + dN[v] == S_all (edge is on an overall shortest 1-N path)
        if dist1[u] != infinity and distN[v] != infinity and \
           dist1[u] + c == dist1[v] and \
           dist1[u] + c + distN[v] == S_all:
            
            # Product P1_prime[u] * PN_prime[v] (capped at 2)
            # Capped product logic: if P1_prime[u]==0 or PN_prime[v]==0, product is 0.
            # Else if P1_prime[u]==1 and PN_prime[v]==1, product is 1.
            # Else (at least one is 2, or both are 1 giving sum >=2 if from different branches, but here it's multiplicative structure) product is 2.
            # This simplifies to min(2, P1_prime[u] * PN_prime[v]) assuming P1/PN values are {0,1,2}.
            product_val = P1_prime[u] * PN_prime[v] 
            product_capped = min(2, product_val)

            if product_capped == total_shortest_paths_count_capped:
                ans_for_this_edge = "Yes"
        
        # Check direction v -> u (if not already determined critical by u->v)
        if ans_for_this_edge == "No" and \
           dist1[v] != infinity and distN[u] != infinity and \
           dist1[v] + c == dist1[u] and \
           dist1[v] + c + distN[u] == S_all:
            
            product_val = P1_prime[v] * PN_prime[u]
            product_capped = min(2, product_val)

            if product_capped == total_shortest_paths_count_capped:
                ans_for_this_edge = "Yes"
        
        results.append(ans_for_this_edge)
            
    sys.stdout.write('
'.join(results) + '
')

solve()