def main():
    import sys
    sys.setrecursionlimit(10**6)
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    
    # Build tree (1-indexed)
    adj = [[] for _ in range(N+1)]
    deg = [0]*(N+1)
    for _ in range(N-1):
        u = int(next(it)); v = int(next(it))
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1

    # allowed[v] is True if v is “allowed” i.e. degree is either 2 or 3.
    # And isT[v] is True if v has degree exactly 2.
    allowed = [False]*(N+1)
    isT = [False]*(N+1)
    for v in range(1, N+1):
        if deg[v] == 2 or deg[v] == 3:
            allowed[v] = True
            if deg[v] == 2:
                isT[v] = True

    # We now want to “follow” a chain starting from a T vertex.
    # In any valid chain the very first neighbor (from a T vertex) must be an allowed vertex that is not T,
    # i.e. it must be an X vertex (which by definition means deg==3).
    #
    # Define a DFS function that when called as dfs(cur, par) returns the count (an integer)
    # of T endpoints reached from ‘cur’ (without going back to par), following only allowed vertices.
    # Note: if the current vertex is T then we have reached an endpoint – return 1 (and do not continue).
    #
    # We use memoization keyed by (cur, par) so that every branch is computed only once.
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def dfs(cur, par):
        # If we are at a T vertex (degree==2) then we are at an endpoint.
        if isT[cur]:
            return 1
        total = 0
        for nxt in adj[cur]:
            if nxt == par:
                continue
            if not allowed[nxt]:
                continue
            total += dfs(nxt, cur)
        return total

    # Now, for every T vertex we “start” a DFS along each allowed neighbor that is not T.
    # (If a T vertex were adjacent to another T vertex the chain would have length=1, 
    # but that would make a 2‐vertex “cycle” – not allowed in a simple graph.)
    global_sum = 0
    for u in range(1, N+1):
        if not isT[u]:
            continue
        for v in adj[u]:
            if not allowed[v]:
                continue
            if isT[v]:
                continue  # avoid direct T--T neighbor (cycle of length 2)
            global_sum += dfs(v, u)
    # Every valid chain (T ... T) is counted twice (once from each endpoint)
    ans = global_sum // 2
    sys.stdout.write(str(ans) + "
")
    
if __name__ == '__main__':
    main()