def main():
    import sys
    sys.setrecursionlimit(3000000)
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    # Build the tree (1-indexed) and compute degrees.
    adj = [[] for _ in range(n+1)]
    deg = [0]*(n+1)
    for _ in range(n-1):
        u = int(next(it))
        v = int(next(it))
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1

    # We want a DFS that, when started from a degree3 vertex v (with the parent being the last vertex,
    # which is a degree2 vertex) follows only vertices with degree 3 and stops when reaching a vertex
    # with degree 2 (which becomes a candidate endpoint).
    memo = {}
    def get_endpoints(v, parent):
        key = (v, parent)
        if key in memo:
            return memo[key]
        endpoints = []
        for w in adj[v]:
            if w == parent:
                continue
            if deg[w] == 2:
                # Found a candidate endpoint (but note: u--w is a tree edge if parent is degree2,
                # so that branch is not allowed; however, since we call this only from a branch where
                # the degree2 starting vertex u is “pushed” into a degree3 neighbor, this situation does
                # not occur).
                endpoints.append(w)
            elif deg[w] == 3:
                endpoints.extend(get_endpoints(w, v))
            # If w's degree is not 2 or 3 then we cannot continue the chain.
        memo[key] = endpoints
        return endpoints

    total = 0
    # For every tree vertex that is degree 2, try launching the search along each edge that goes to a degree3 vertex.
    # (We skip neighbors of degree2 because then u would be adjacent to that neighbor in the tree.)
    for u in range(1, n+1):
        if deg[u] == 2:
            for w in adj[u]:
                if deg[w] == 3:
                    endpoints = get_endpoints(w, u)
                    for other in endpoints:
                        # Ensure a proper cycle (at least one degree3 vertex in between) and count each pair once.
                        if other != u and u < other:
                            total += 1
    sys.stdout.write(str(total))
    
if __name__ == '__main__':
    main()