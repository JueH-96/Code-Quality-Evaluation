def main():
    import sys,sys
    sys.setrecursionlimit(3000000)
    data = sys.stdin.buffer.read().split()
    if not data: 
        return
    it = iter(data)
    n = int(next(it))
    # Build undirected graph.
    g = [[] for _ in range(n+1)]
    for _ in range(n-1):
        u = int(next(it)); v = int(next(it)); w = int(next(it))
        g[u].append((v, w))
        g[v].append((u, w))
        
    # We root the tree at 1.
    parent = [0]*(n+1)   # parent[v]
    d = [0]*(n+1)        # d[v]: distance from 1
    tot = [0]*(n+1)      # tot[v]: sum of weights in subtree v
    Mval = [0]*(n+1)     # Mval[v]: maximum d[u] for u in subtree of v
    children = [[] for _ in range(n+1)]
    
    # Do a DFS (iterative helps avoid recursion‐limit issues in deep trees)
    sys.setrecursionlimit(3000000)
    stack = [1]
    parent[1] = 0
    d[1] = 0
    order = []   # will hold the order for post‐order processing
    visited = [False]*(n+1)
    visited[1] = True
    while stack:
        u = stack.pop()
        order.append(u)
        for v,w in g[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                d[v] = d[u] + w
                children[u].append(v)
                stack.append(v)
    # Process nodes in reverse order (postorder) to compute tot[v] and Mval[v]
    for u in reversed(order):
        tot[u] = 0
        Mval[u] = d[u]
        for v in children[u]:
            tot[u] += (d[v]-d[u]) + tot[v]   # (edge weight from u,v) + tot[v]
            if Mval[v] > Mval[u]:
                Mval[u] = Mval[v]
                
    total = tot[1]    # total is the sum of all edge lengths
    # Find vertex v_max with maximum d[v]
    v_max = 1
    for i in range(1, n+1):
        if d[i] > d[v_max]:
            v_max = i
    base = d[v_max]   # the “base” union is d(v_max)
    
    # Reconstruct the unique 1-v_max path.
    P = []
    cur = v_max
    while cur:
        P.append(cur)
        cur = parent[cur]
    P.reverse()
    inP = [False]*(n+1)
    for v in P:
        inP[v] = True

    # Now every branch that is “attached” to the main path is represented
    # by any vertex v (v≠1) that is not in P but has parent in P.
    # For such a branch (with attachment p), let:
    #    candidate₁ = Mval[v] – d[p]   (best union achievable in that branch with one pick)
    #    full_val   = (d[v]-d[p]) + tot[v]  (if the branch is fully covered)
    #    candidate₂ = full_val – candidate₁   (extra gain if an extra pick is used on that branch)
    # (Note that if candidate₂ is zero then one pick already “covers” the branch as far as possible.)
    pool = []
    for v in range(2, n+1):
        if (not inP[v]) and inP[parent[v]]:
            p = parent[v]
            cand1 = Mval[v] - d[p]
            full_val = (d[v] - d[p]) + tot[v]
            cand2 = full_val - cand1
            pool.append(cand1)
            if cand2 > 0:
                pool.append(cand2)
    pool.sort(reverse=True)
    pref = [0]
    for val in pool:
        pref.append(pref[-1] + val)
    pool_len = len(pool)
    maxExtra = total - base  # maximum extra union achievable
    out_lines = []
    # For each K=1,...,n, if we can use extra = (K-1) picks to cover branches,
    # then the union = base + (prefix sum of the top (K-1) values in pool)
    # (capped by maxExtra). The answer printed is 2×union.
    for k in range(1, n+1):
        extraPicks = k - 1
        if extraPicks > pool_len:
            extraPicks = pool_len
        extra_val = pref[extraPicks]
        if extra_val > maxExtra:
            extra_val = maxExtra
        union_val = base + extra_val
        if union_val > total:
            union_val = total
        ans = union_val * 2
        out_lines.append(str(ans))
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()