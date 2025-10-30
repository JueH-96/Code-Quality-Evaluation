def main():
    import sys
    sys.setrecursionlimit(300000)
    data = sys.stdin.read().strip().split()
    if not data: 
        return
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    total = N * K
    # Build the tree (1-indexed)
    adj = [[] for _ in range(total+1)]
    for _ in range(total - 1):
        u = int(next(it))
        v = int(next(it))
        adj[u].append(v)
        adj[v].append(u)
    
    # Special case: if K == 1, every vertex stands alone.
    if K == 1:
        sys.stdout.write("Yes")
        return

    # We'll perform DFS.
    # The DFS function returns:
    #   > 0 : the open chain length from that node (chain length counts number of vertices in the open chain)
    #   0   : node is "closed" (u is consumed as part of a complete path)
    #  -1   : failure (no valid decomposition)
    def dfs(u, parent):
        cand = []  # candidate open chain lengths from children
        for v in adj[u]:
            if v == parent:
                continue
            res = dfs(v, u)
            if res == -1:
                return -1
            if res > 0:
                cand.append(res)
        # u can only help at most two open chains.
        if len(cand) > 2:
            return -1
        if len(cand) == 2:
            a, b = cand[0], cand[1]
            # If adding u (which contributes +1) makes the total K vertices then these two chains merge.
            if a + b + 1 == K:
                return 0  # completed a path at u; nothing propagates upward.
            else:
                return -1
        if len(cand) == 1:
            new_len = cand[0] + 1  # extend that open chain with u.
            if new_len == K:
                # the chain completes exactly here
                return 0
            elif new_len < K:
                return new_len
            else:
                return -1  # should not happen.
        # No candidate from children: we start a chain at u.
        return 1

    res = dfs(1, -1)
    # In order to cover all vertices, the open chain must be completed (closed) at the root.
    if res == 0:
        sys.stdout.write("Yes")
    else:
        sys.stdout.write("No")
 
if __name__ == '__main__':
    main()