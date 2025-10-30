def main():
    import sys
    input = sys.stdin.readline
    from itertools import permutations
    
    INF = 10**15
    
    # Read N (islands) and M (bridges)
    N, M = map(int, input().split())
    
    # Initialize distance matrix for Floyd-Warshall
    dist = [[INF]*N for _ in range(N)]
    for i in range(N):
        dist[i][i] = 0
    
    # Read all edges
    edges = []
    for _ in range(M):
        u, v, t = map(int, input().split())
        u -= 1
        v -= 1
        edges.append((u, v, t))
        # Possibly multiple parallel edges, keep the minimum
        if t < dist[u][v]:
            dist[u][v] = t
            dist[v][u] = t
    
    # Floyd-Warshall to get all-pairs shortest paths
    # N <= 400, so O(N^3) = 64e6 in the worst case, which can be borderline but is
    # typically doable in optimized Python or with sufficient time limits.
    for k in range(N):
        distk = dist[k]
        for i in range(N):
            rowi = dist[i]
            ik = rowi[k]
            for j in range(N):
                alt = ik + distk[j]
                if rowi[j] > alt:
                    rowi[j] = alt
    
    # Precompute permutations for k=0..5
    # perms_for_k[k] = list of all permutations of range(k)
    from math import factorial
    perms_for_k = [[] for _ in range(6)]
    for k in range(6):
        if k == 0:
            perms_for_k[k] = [()]  # one "empty" permutation
        else:
            perms_for_k[k] = list(permutations(range(k)))
    
    # Read Q (number of queries)
    Q = int(input())
    
    # Process each query
    out = []
    for _ in range(Q):
        K = int(input())
        required = list(map(int, input().split()))
        
        # Gather the (u, v, t) for the K required edges
        eTC = [edges[idx - 1] for idx in required]  # zero-based in edges
        
        best = INF
        
        # Try all permutations of these K edges
        for perm in perms_for_k[K]:
            # For each permutation, try all 2^K ways of choosing directions
            # (0 => use edge u->v, 1 => use edge v->u)
            # We will "force" exactly one crossing of each required edge in order,
            # but note the intermediate shortest paths may also use any edges
            # (including these again) if beneficial.
            for dirs in range(1 << K):
                cost = 0
                cur = 0  # start from island 1 => index 0
                # Cross the edges in the chosen order/directions
                for i in range(K):
                    (u, v, t) = eTC[perm[i]]
                    if ((dirs >> i) & 1) == 0:
                        # go cur -> u, cross edge u->v
                        cost += dist[cur][u]
                        cost += t
                        cur = v
                    else:
                        # go cur -> v, cross edge v->u
                        cost += dist[cur][v]
                        cost += t
                        cur = u
                # finally go cur -> N-1
                cost += dist[cur][N-1]
                if cost < best:
                    best = cost
        
        out.append(str(best))
    
    print("
".join(out))

if __name__ == "__main__":
    main()