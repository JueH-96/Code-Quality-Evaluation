def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    K = int(next(it))
    
    # Shortcut: if K == 1 then every sum mod 1 is 0.
    if K == 1:
        print(0)
        return

    edges = []
    for _ in range(M):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        w = int(next(it))
        edges.append((w, u, v))
    
    # We sort edges by weight. This does not change the final answer
    # because we are enumerating all spanning trees (each tree is built with
    # a strictly increasing index order from the sorted list).
    edges.sort(key=lambda x: x[0])
    
    best = K  # The minimum mod value possible is 0 and maximum is K-1.

    # Simple union-find (since N is small, we can avoid heavy optimizations)
    def find(x, parent):
        while x != parent[x]:
            x = parent[x]
        return x

    # We'll use DFS to enumerate all spanning trees.
    # 'start' is the current edge index in edges list,
    # 'count' is how many edges have been chosen so far,
    # 'current_sum' is the sum of the chosen edges and
    # 'parent' is the union-find parent list.
    def dfs(start, count, current_sum, parent):
        nonlocal best
        # If we've chosen exactly N-1 edges, we have a spanning tree.
        if count == N - 1:
            mod_val = current_sum % K
            if mod_val < best:
                best = mod_val
            return
        if best == 0:
            # Cannot improve on 0.
            return
        # Number of edges needed to complete a spanning tree.
        rem_needed = (N - 1) - count
        # If not enough edges remain, we prune.
        if M - start < rem_needed:
            return
        
        for j in range(start, M):
            if M - j < rem_needed:
                break
            w, u, v = edges[j]
            ru = find(u, parent)
            rv = find(v, parent)
            if ru == rv:
                continue
            # Copy the union-find structure and union the two sets.
            new_parent = parent[:]  # shallow copy is fine since N is small
            new_parent[rv] = ru
            dfs(j + 1, count + 1, current_sum + w, new_parent)
            if best == 0:
                return

    # Initialize union-find: each vertex is its own parent.
    initial_parent = list(range(N))
    dfs(0, 0, 0, initial_parent)
    print(best)

    
if __name__ == '__main__':
    main()