def main():
    import sys, itertools
    input = sys.stdin.readline
    INF = 10**18
    
    # read basic graph properties
    N, M = map(int, input().split())
    
    # store the bridges (convert islands to 0-indexed)
    bridges = []
    # We also build a matrix for Floyd Warshall
    d = [[INF]*N for _ in range(N)]
    for i in range(N):
        d[i][i] = 0
        
    # read bridges and update matrix
    U = [0]*M
    V = [0]*M
    T_list = [0]*M
    for i in range(M):
        u,v,t = map(int, input().split())
        u -= 1
        v -= 1
        U[i] = u
        V[i] = v
        T_list[i] = t
        if t < d[u][v]:
            d[u][v] = t
            d[v][u] = t
        else:
            # even if there is already a better edge, we still record this bridge's info
            pass
        bridges.append((u, v, t))
    
    # Floyd Warshall to compute all pairs shortest distances.
    # Since N is 400, this triple loop (O(N^3)) is acceptable.
    for k in range(N):
        dk = d[k]
        for i in range(N):
            di = d[i]
            alt = di[k]
            if alt == INF: continue
            for j in range(N):
                nd = alt + dk[j]
                if nd < di[j]:
                    di[j] = nd

    Q = int(input())
    results = []
    for _ in range(Q):
        k = int(input())
        # list of required bridge indices (convert to 0-indexed)
        req_bridges = list(map(lambda x: int(x)-1, input().split()))
        # We will build lists for endpoints & cost.
        # For a given required bridge, we denote its endpoints as (a, b) and cost t.
        req = []
        for idx in req_bridges:
            a = U[idx]
            b = V[idx]
            t = T_list[idx]
            req.append((a, b, t))
        # Now, we want to “choose” an order (permutation) and orientation for the required bridges.
        ans = INF
        indices = list(range(k))
        # iterate over all orders (permutations)
        for order in itertools.permutations(indices):
            # there are 2^k orientation choices
            # We represent orientation by a bitmask: if bit==0 then we traverse using (a -> b), if bit==1 then (b -> a)
            for mask in range(1 << k):
                cost = 0
                cur = 0  # starting island: 1 is index 0
                possible = True
                for pos in range(k):
                    i = order[pos]
                    a,b,t = req[i]
                    if (mask >> pos) & 1:
                        # use the bridge in reverse: from b to a
                        start, end = b, a
                    else:
                        start, end = a, b
                    # travel from current island to the start endpoint of the required bridge
                    if d[cur][start] == INF:
                        possible = False
                        break
                    cost += d[cur][start]
                    # cross the bridge (remember cost is the explicit cost given).
                    cost += t
                    cur = end
                if not possible:
                    continue
                if d[cur][N-1] == INF:
                    continue
                cost += d[cur][N-1]
                if cost < ans:
                    ans = cost
        results.append(ans)
    # output the results
    sys.stdout.write("
".join(str(x) for x in results))
    
if __name__ == '__main__':
    main()