def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Parsing the first line: N, M
    N, M = map(int, input_data[:2])
    # Next M lines: U_i, V_i, T_i
    edges = []
    idx = 2
    for _ in range(M):
        u, v, t = map(int, input_data[idx:idx+3])
        idx += 3
        # Convert to 0-based for easier handling
        edges.append((u-1, v-1, t))
    
    # Prepare a matrix for all-pairs shortest paths
    INF = 10**18
    dist = [[INF]*N for _ in range(N)]
    for i in range(N):
        dist[i][i] = 0
    
    # Fill in direct connections (there may be multiple bridges between two islands)
    for (u, v, t) in edges:
        if t < dist[u][v]:
            dist[u][v] = t
        if t < dist[v][u]:
            dist[v][u] = t
    
    # Floyd-Warshall for all-pairs shortest path (N <= 400)
    for k in range(N):
        dk = dist[k]
        for i in range(N):
            dik = dist[i][k]
            row_i = dist[i]
            # Unrolling the j-loop may help a bit in performance, but let's keep it standard
            for j in range(N):
                alt = dik + dk[j]
                if alt < row_i[j]:
                    row_i[j] = alt
    
    # Now handle queries
    Q = int(input_data[idx]); idx += 1
    
    # Precompute factorial permutations up to K=5
    # We'll just use itertools each time, as K <= 5 is small.
    from itertools import permutations
    
    out = []
    for _ in range(Q):
        K = int(input_data[idx])
        idx += 1
        # The K edges that must be used
        query_edges_indices = list(map(int, input_data[idx:idx+K]))
        idx += K
        
        # Gather (u,v,t) for these edges (0-based)
        req = [edges[eid-1] for eid in query_edges_indices]
        
        # If K=1, we can just do two directions easily; but we'll use the same approach for uniformity
        
        # We'll try all permutations of these K edges
        best_cost = INF
        
        if K == 1:
            # Slight shortcut: only 2 direction checks
            (u, v, t) = req[0]
            # direction 0: 1->u + t + v->N
            cost1 = dist[0][u] + t + dist[v][N-1]
            # direction 1: 1->v + t + u->N
            cost2 = dist[0][v] + t + dist[u][N-1]
            best_cost = min(cost1, cost2)
        else:
            # For general K (up to 5)
            # Try all permutations p of range(K)
            # And for each permutation, try all bitmasks for directions
            # direction bit for edge i in permutation => 0 means (u->v), 1 means (v->u)
            for p in permutations(range(K)):
                # We'll do a bit-mask 0..(1<<K)-1 for directions
                for mask in range(1 << K):
                    cur_cost = 0
                    # current node = 0 (island 1 in 0-based)
                    current_node = 0
                    feasible = True
                    for i in range(K):
                        (u, v, t) = req[p[i]]
                        dbit = (mask >> i) & 1
                        if dbit == 0:
                            # crossing u->v
                            c = dist[current_node][u]
                            if c == INF:
                                feasible = False
                                break
                            cur_cost += c + t
                            current_node = v
                        else:
                            # crossing v->u
                            c = dist[current_node][v]
                            if c == INF:
                                feasible = False
                                break
                            cur_cost += c + t
                            current_node = u
                        if cur_cost >= best_cost:  # small pruning
                            feasible = False
                            break
                    
                    if feasible:
                        # Finally go from current_node to N-1
                        c = dist[current_node][N-1]
                        if c != INF:
                            cur_cost += c
                            if cur_cost < best_cost:
                                best_cost = cur_cost
            
        out.append(str(best_cost))
    
    print("
".join(out))

# Call main() to ensure we run the solution.
main()