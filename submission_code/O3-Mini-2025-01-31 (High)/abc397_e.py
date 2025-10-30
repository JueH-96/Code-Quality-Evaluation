def main():
    import sys,sys
    sys.setrecursionlimit(500000)
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    try:
        N = int(next(it))
        K = int(next(it))
    except StopIteration:
        return
    total_vertices = N * K

    # Build the tree (vertices 1...NK)
    adj = [[] for _ in range(total_vertices+1)]
    for _ in range(total_vertices - 1):
        try:
            u = int(next(it))
            v = int(next(it))
        except StopIteration:
            break
        adj[u].append(v)
        adj[v].append(u)
        
    # Special-case: if K == 1, every vertex is its own path.
    if K == 1:
        sys.stdout.write("Yes")
        return

    # DFS: for each vertex v (with parent par), we return one of:
    #   an integer x in [1, K-1] meaning “free chain” length from a descendant up to v,
    #   None meaning that v has been "used" (the block completed in the subtree),
    #   or -1 if an inconsistency is found.
    def dfs(v, par):
        free_list = []  # free chain values from children
        for w in adj[v]:
            if w == par:
                continue
            state = dfs(w, v)
            if state == -1:
                return -1
            if state is not None:
                free_list.append(state)
        if len(free_list) > 2:
            return -1
        if len(free_list) == 2:
            # They must sum to K-1 so that (a+b+1) = K.
            if free_list[0] + free_list[1] != K - 1:
                return -1
            return None
        if len(free_list) == 1:
            candidate = free_list[0] + 1
            if candidate < K:
                return candidate
            elif candidate == K:
                return None
            else:
                return -1
        # No free child? Start a new chain here.
        return 1

    res = dfs(1, 0)
    # For a valid complete partition the free chain must be completed (state None) at the root.
    if res == -1 or res is not None:
        sys.stdout.write("No")
    else:
        sys.stdout.write("Yes")
        
if __name__ == '__main__':
    main()