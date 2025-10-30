def main():
    import sys,sys
    sys.setrecursionlimit(500000)
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    # Build tree (1-indexed)
    adj = [[] for _ in range(n+1)]
    tot_weight = 0
    for _ in range(n-1):
        u = int(next(it)); v = int(next(it)); w = int(next(it))
        adj[u].append((v,w))
        adj[v].append((u,w))
        tot_weight += w

    # For each vertex v (in the DFS tree rooted at 1)
    # h[v] = maximum distance from v downwards (not including the parent edge)
    # tot_sub[v] = sum of weights of all edges in the subtree rooted at v
    h = [0]*(n+1)
    tot_sub = [0]*(n+1)
    parent = [0]*(n+1)
    def dfs(v, par):
        parent[v] = par
        maxh = 0
        sumtot = 0
        for nxt, w in adj[v]:
            if nxt == par:
                continue
            dfs(nxt, v)
            cand = w + h[nxt]
            if cand > maxh:
                maxh = cand
            sumtot += w + tot_sub[nxt]
        h[v] = maxh
        tot_sub[v] = sumtot
    dfs(1,0)

    # Decompose the tree by the branches from the root.
    # For every direct neighbor u of 1:
    #   best = L(1,u) + h[u]
    #   full = L(1,u) + tot_sub[u]
    #   extra = full - best
    baseList = []  # one best per branch
    extraList = [] # and its corresponding extra
    fullSum = 0    # sum over all branches = full union = total forced union if all branches are fully forced.
    for nxt, w in adj[1]:
        best = w + h[nxt]
        fullV = w + tot_sub[nxt]
        extra = fullV - best  # = tot_sub[nxt] - h[nxt]
        baseList.append(best)
        extraList.append(extra)
        fullSum += fullV
    # r = number of branches (children of 1)
    r = len(baseList)
    # sort descending:
    baseList.sort(reverse=True)
    extraList.sort(reverse=True)
    prefBase = [0]
    for x in baseList:
        prefBase.append(prefBase[-1] + x)
    prefExtra = [0]
    for x in extraList:
        prefExtra.append(prefExtra[-1] + x)
    
    # Now answer for each K = 1,..., n.
    # (Recall: if U is the forced union, then Takahashi’s route cost = 2*U.)
    out_lines = []
    for K in range(1, n+1):
        if K <= r:
            U = prefBase[K]
        elif K <= 2 * r:
            U = prefBase[r] + prefExtra[K - r]
        else:
            U = fullSum
        out_lines.append(str(2 * U))
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()