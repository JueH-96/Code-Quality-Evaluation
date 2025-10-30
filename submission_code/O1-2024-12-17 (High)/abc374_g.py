def main():
    import sys
    sys.setrecursionlimit(10**7)

    N = int(sys.stdin.readline())
    # Adjacency for the directed graph of letters (0..25 for A..Z).
    adj = [[] for _ in range(26)]
    # Undirected adjacency just to find connected components (ignoring direction).
    undir_adj = [[] for _ in range(26)]
    used = [False]*26  # Track which letters actually appear.

    # Read edges (two-letter names) and build graph
    for _ in range(N):
        s = sys.stdin.readline().strip()
        x = ord(s[0]) - ord('A')
        y = ord(s[1]) - ord('A')
        used[x] = True
        used[y] = True
        adj[x].append(y)
        # For connected components ignoring direction:
        undir_adj[x].append(y)
        undir_adj[y].append(x)

    # Find connected components (ignoring direction)
    comp_id = [-1]*26
    comp_count = 0

    def dfs_mark(start):
        stack = [start]
        comp_id[start] = comp_count
        while stack:
            v = stack.pop()
            for w in undir_adj[v]:
                if used[w] and comp_id[w] == -1:
                    comp_id[w] = comp_count
                    stack.append(w)

    for v in range(26):
        if used[v] and comp_id[v] == -1:
            dfs_mark(v)
            comp_count += 1

    # Group vertices by component
    comp_vertices = [[] for _ in range(comp_count)]
    for v in range(26):
        if used[v]:
            comp_vertices[comp_id[v]].append(v)

    # Tarjan SCC for each connected component
    def tarjan_scc(subverts, subadj):
        """
        subverts: list of actual letter-IDs in this connected component
        subadj[u] adjacency among indices [0..len(subverts)-1],
                 where u is an index into subverts.
        Returns (scc_id, scc_count):
          scc_id[u] is which SCC index the node u belongs to
          scc_count is total number of SCCs found
        """
        n = len(subverts)
        stack = []
        on_stack = [False]*n
        index = [-1]*n
        lowlink = [0]*n
        current_index = 0
        scc_id = [-1]*n
        scc_count = 0

        def strongconnect(u):
            nonlocal current_index, scc_count
            index[u] = current_index
            lowlink[u] = current_index
            current_index += 1
            stack.append(u)
            on_stack[u] = True

            for w in subadj[u]:
                if index[w] == -1:
                    strongconnect(w)
                    lowlink[u] = min(lowlink[u], lowlink[w])
                elif on_stack[w]:
                    lowlink[u] = min(lowlink[u], index[w])

            # If u is a root of an SCC
            if lowlink[u] == index[u]:
                c = scc_count
                scc_count += 1
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc_id[w] = c
                    if w == u:
                        break

        for i in range(n):
            if index[i] == -1:
                strongconnect(i)

        return scc_id, scc_count

    answer = 0

    # Process each undirected-connected component separately
    for cid in range(comp_count):
        verts = comp_vertices[cid]
        if not verts:
            # Should not happen unless something is unused
            continue

        # Build adjacency restricted to this component
        # First map each letter in this component to a local index
        local_index = {}
        for i, v in enumerate(verts):
            local_index[v] = i

        # subadj[u] = list of local indices for adjacency
        nloc = len(verts)
        subadj = [[] for _ in range(nloc)]
        for v in verts:
            vloc = local_index[v]
            for w in adj[v]:
                if comp_id[w] == cid:  # same undirected component
                    subadj[vloc].append(local_index[w])

        # Run Tarjan to find SCCs inside this component
        scc_id, scc_count = tarjan_scc(verts, subadj)

        # If there's only 1 SCC, it could still have internal edges;
        # but let's build the condensation "graph" of these SCCs anyway.
        # Count how many edges go from SCC X to SCC Y (X != Y).
        e_count = {}
        for u in range(nloc):
            X = scc_id[u]
            for w in subadj[u]:
                Y = scc_id[w]
                if X != Y:
                    e_count[(X, Y)] = e_count.get((X, Y), 0) + 1

        # Compute outdegree of each SCC in the condensation
        outdeg = [0]*scc_count
        for (X, Y), cnt in e_count.items():
            outdeg[X] += 1

        # Count how many SCCs are sinks (outdeg=0).
        sink_count = sum(1 for x in range(scc_count) if outdeg[x] == 0)

        # L = max number of cross-edges from X->Y, if any exist
        if e_count:
            L = max(e_count.values())
        else:
            L = 0

        # The formula for needed strings in this component:
        #   max(sink_count, L)
        # Because each sink SCC must be visited by at least one path,
        # and each cross-edge X->Y must appear in distinct paths if it has multiplicity > 1.
        # (In our problem, at most 1 edge per letter-pair, but if multiple letter-pairs
        #  connect the same SCCs, that increments e_count, so effectively L can be >1.)
        sub_ans = max(sink_count, L)
        answer += sub_ans

    print(answer)