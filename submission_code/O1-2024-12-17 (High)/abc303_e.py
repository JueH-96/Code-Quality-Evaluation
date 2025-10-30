def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    edges = input_data[1:]
    adj = [[] for _ in range(N+1)]
    deg = [0]*(N+1)
    
    idx = 0
    for _ in range(N-1):
        u = int(edges[idx]); v = int(edges[idx+1])
        idx += 2
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1

    # We call any vertex with degree >= 3 a "forced center":
    # it must be the center of some star (and its level = its degree).
    # Vertices of degree 1 or 2 will either be leaves in those high-degree stars
    # or else form their own level-2 stars in chains.
    forced_center = [False]*(N+1)
    for v in range(1, N+1):
        if deg[v] >= 3:
            forced_center[v] = True

    visited = [False]*(N+1)
    centers = []
    # First collect all forced centers (degree >= 3).
    # Their "level" will simply be deg[v].
    forced_centers_list = []
    for v in range(1, N+1):
        if forced_center[v]:
            forced_centers_list.append(v)

    # Build a graph of the "non-forced" vertices (deg 1 or 2),
    # ignoring the forced-center vertices.  Each connected component
    # in that subgraph must be a simple path (no branching),
    # because any branch would have created a forced center (deg>=3)
    # in the original tree.
    #
    # We will find these path-components by a DFS/BFS, then break each path
    # into segments of length 3, placing a level-2 star center in the middle
    # of each triple.
    
    # We will store adjacency for the subgraph of non-forced centers:
    # but easier is just to do a BFS/DFS with a condition "skip forced centers."
    sys.setrecursionlimit(10**7)
    
    sub_adj = [[] for _ in range(N+1)]  # adjacency restricted to non-forced
    for v in range(1, N+1):
        if not forced_center[v]:
            for w in adj[v]:
                if not forced_center[w]:
                    sub_adj[v].append(w)
    
    visited = [False]*(N+1)

    def get_component(start):
        """Return all vertices in the connected component of 'start'
           (ignoring forced centers), via a DFS/BFS."""
        stack = [start]
        comp = []
        visited[start] = True
        while stack:
            node = stack.pop()
            comp.append(node)
            for nx in sub_adj[node]:
                if not visited[nx]:
                    visited[nx] = True
                    stack.append(nx)
        return comp

    # We'll also build a small helper to find endpoints of the path:
    # In this subgraph, each vertex can have subgraph-degree 1 or 2
    # (0 if it's an isolated vertex, which should not happen in a valid final).
    # If it's a path, it should have exactly two endpoints with degree=1,
    # unless the path length < 2 (which the problem statement disallows,
    # given k>=2 for each star).
    
    def build_path(component):
        # Build adjacency for the *component only* among the subgraph,
        # find its endpoints, then list the path from one endpoint to the other.
        cadj = [[] for _ in range(len(component))]
        index_of = {}
        for i, v in enumerate(component):
            index_of[v] = i

        for v in component:
            i = index_of[v]
            for w in sub_adj[v]:
                cadj[i].append(index_of[w])

        # find endpoints (degree=1 in cadj)
        deg_sub = [len(cadj[i]) for i in range(len(component))]
        endpoints = [i for i in range(len(component)) if deg_sub[i] == 1]
        # The problem statement guarantees a valid decomposition,
        # so we expect exactly 2 endpoints if length >= 2,
        # or possibly 0 endpoints if there's only 1 vertex (which can't form k>=2 star).
        if len(endpoints) != 2:
            # theoretically this should not happen in a valid input
            # (unless there's only 1 vertex -- not valid for k>=2),
            # or a bigger mismatch. But we trust input validity.
            return []

        # Let's do a simple path-walk from endpoints[0] to endpoints[1].
        ep1, ep2 = endpoints[0], endpoints[1]
        path = []
        cur = ep1
        parent = -1
        while True:
            path.append(component[cur])
            if cur == ep2:
                break
            # find next
            nxt = -1
            for w in cadj[cur]:
                if w != parent:
                    nxt = w
                    break
            parent = cur
            cur = nxt
        return path

    # For each path, we partition it in triples.  The middle of each triple
    # is a deg=2 vertex - that becomes a new center of level=2.
    
    # We'll gather these new centers in a list:
    new_centers = []
    
    for v in range(1, N+1):
        if (not forced_center[v]) and (not visited[v]):
            comp = get_component(v)
            if len(comp) == 1:
                # single-vertex component => can't form a star of k>=2
                # should not happen in a valid final. We skip or ignore.
                continue
            path = build_path(comp)
            # Now tile this path in consecutive triples
            # The length must be multiple of 3, guaranteed by the problem.
            # If not, the input wouldn't come from a valid star arrangement.
            Lp = len(path)
            # We place star centers at path[i+1] for i=0,3,6,... 
            # so each triple is (path[i], path[i+1], path[i+2]).
            for i in range(0, Lp, 3):
                c = path[i+1]
                new_centers.append(c)
    
    # Combine forced centers and newly found deg=2 centers:
    all_centers = forced_centers_list + new_centers
    # Compute the level = degree in T for each center:
    levels = [deg[c] for c in all_centers]
    levels.sort()
    print(" ".join(map(str, levels)))


# Don't forget to call main().
if __name__ == "__main__":
    main()