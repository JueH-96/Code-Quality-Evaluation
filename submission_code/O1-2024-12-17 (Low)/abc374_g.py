def main():
    import sys
    sys.setrecursionlimit(10**7)

    N = int(sys.stdin.readline().strip())
    edges = []
    used = [[False]*26 for _ in range(26)]  # used[x][y] = True means chr(x+'A') -> chr(y+'A') is in U

    for _ in range(N):
        s = sys.stdin.readline().strip()
        v = ord(s[0]) - ord('A')
        w = ord(s[1]) - ord('A')
        used[v][w] = True

    # Build adjacency for the directed graph of size 26
    graph = [[] for _ in range(26)]
    for v in range(26):
        for w in range(26):
            if used[v][w]:
                graph[v].append(w)

    # To solve the problem, we note that each "NG-list string" must be
    # a walk in this graph (because every adjacent pair of letters must be in U).
    # We need to cover all used edges at least once across our chosen walks.
    #
    # However, we are allowed to reuse edges.  This means within any *strongly connected*
    # portion of the graph, we can collect all edges using a single walk (since we can loop
    # around as needed).  Between different strongly connected components (SCCs), we can
    # chain coverage via edges that connect them if the direction in the condensation
    # graph permits it.
    #
    # Ultimately, one finds that the minimum number of such walks needed equals
    #   sum over each "connected component (ignoring direction) of the condensation graph"
    #     of  max(#sources, #sinks)
    # where #sources is the count of SCC-nodes in that component with in-degree = 0
    # (within that component), and #sinks is the count of SCC-nodes with out-degree = 0.
    #
    # Implementation outline:
    # 1) Run SCC (Tarjan or Kosaraju) on the 26-vertex graph to get scc_id[v] for each v.
    # 2) Build the condensation DAG: for each edge v->w with scc_id[v]!=scc_id[w],
    #    add an edge scc_id[v]->scc_id[w] in the "scc_graph".
    # 3) Find connected components (undirected) of that scc_graph.  For each such component,
    #    compute how many of its nodes have in_degree=0 and out_degree=0 (within that sub-DAG).
    #    The contribution from that component = max(#sources, #sinks).  Sum these up.

    # Step 1) Find SCCs for the graph of size 26.
    # We'll use Tarjan's algorithm.

    NVERT = 26
    scc_id = [-1]*NVERT
    lowlink = [0]*NVERT
    depth = [-1]*NVERT
    stack = []
    in_stack = [False]*NVERT
    current_depth = 0
    scc_count = 0

    def tarjan(v):
        nonlocal current_depth, scc_count
        depth[v] = current_depth
        lowlink[v] = current_depth
        current_depth += 1
        stack.append(v)
        in_stack[v] = True

        for w in graph[v]:
            if depth[w] == -1:
                tarjan(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif in_stack[w]:
                lowlink[v] = min(lowlink[v], depth[w])

        # If v is root of SCC
        if lowlink[v] == depth[v]:
            # Start popping until v
            while True:
                top = stack.pop()
                in_stack[top] = False
                scc_id[top] = scc_count
                if top == v:
                    break
            scc_count += 1

    for v in range(NVERT):
        # Only run for vertices that have edges or might be part of the graph
        # Actually, to be safe, we run Tarjan if it hasn't been visited.
        # Even if a vertex had no edges, we still want it in its own SCC if it
        # appears at all.  But the problem states only used product names matter,
        # though a vertex could appear if it is the first or second letter of some pair.
        if depth[v] == -1:
            # Check if v is relevant (no edges in or out?), we still can tarjan it.
            # Because it could appear as an endpoint in used[].
            # We'll do it unconditionally.
            # But let's see if the vertex is actually used in any edge, in or out:
            in_or_out = False
            for w in range(26):
                if used[v][w] or used[w][v]:
                    in_or_out = True
                    break
            # If in_or_out is True, we do Tarjan to place it in an SCC.
            # If false, it never appears in the used pairs and can be safely
            # ignored. (It won't affect the solution because it doesn't appear
            # in any "NG-list" substring constraints.)
            if in_or_out:
                tarjan(v)

    # scc_count is the total number of strongly connected components
    # scc_id[v] is the ID of the SCC containing v, in [0..scc_count-1]

    # Step 2) Build condensation graph among the scc_count SCCs
    # For convenience, only add edges scc_u -> scc_v if u->w crosses components.
    # We'll track in_degree and out_degree of each SCC-node in that condensation DAG,
    # and we'll build adjacency to figure out its connected components ignoring direction.

    scc_graph = [[] for _ in range(scc_count)]  # adjacency in the DAG
    scc_in_deg = [0]*scc_count
    scc_out_deg = [0]*scc_count

    # We'll also maintain for ignoring-direction adjacency:
    undirected_scc_graph = [[] for _ in range(scc_count)]

    # We only consider edges that connect different SCCs
    for v in range(NVERT):
        if scc_id[v] == -1:
            # means v wasn't actually used in any edge
            continue
        for w in graph[v]:
            if scc_id[w] == -1:
                continue
            if scc_id[v] != scc_id[w]:
                scc_graph[scc_id[v]].append(scc_id[w])
    
    # Build in_deg/out_deg
    # Also build undirected edges for connected-component finding on the condensation DAG
    # (We ignore self-loop edges from scc_i -> scc_i for counting source/sink in this step)
    for u in range(scc_count):
        # remove duplicates by using a set
        nxts = list(set(scc_graph[u]))
        scc_graph[u] = nxts
        for v in nxts:
            scc_out_deg[u] += 1
            scc_in_deg[v] += 1
            undirected_scc_graph[u].append(v)
            undirected_scc_graph[v].append(u)

    # We'll find connected components of the condensation DAG ignoring direction
    visited = [False]*scc_count
    components = []  # list of lists of SCC nodes

    def dfs_cc(start):
        stack = [start]
        comp = []
        visited[start] = True
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nx in undirected_scc_graph[cur]:
                if not visited[nx]:
                    visited[nx] = True
                    stack.append(nx)
        return comp

    for i in range(scc_count):
        if not visited[i]:
            components.append(dfs_cc(i))

    # Now for each component, compute #sources and #sinks
    # A "source" in that sub-DAG means in_degree=0 *within that subcomponent*,
    # likewise for "sink".
    #
    # However, the in_degree/out_degree we've computed is global in the condensation DAG.
    # We only need to consider edges between nodes that lie in the same subcomponent,
    # so we must recalc in/out deg restricted to the subcomponent.

    # We'll store adjacency restricted to the subcomponent for convenience:
    answer = 0
    scc_graph_set = [set(adj) for adj in scc_graph]  # so we can test existence quickly

    for comp in components:
        # If comp has only one node "c", and if it has no edges to itself
        # in this sub-DAG, then #sources=1 and #sinks=1 => answer+=1.
        # Otherwise, we count how many in this subcomponent have no incoming
        # from inside the comp, and how many have no outgoing to inside the comp.
        if len(comp) == 1:
            c = comp[0]
            # Check if there's a self-loop c->c in the condensation DAG
            # Actually, for source/sink counting, a self-loop does not create
            # an incoming edge from a different node. So it does not affect
            # whether it's a source or sink. This single node is simultaneously
            # source and sink if no other edges come in or out from different nodes.
            # We see if in_degree or out_degree from that subcomponent is 0.
            # But let's do the general approach anyway.
            # We'll just do the general approach with the subcomponent's adjacency.

        # Build local in/out deg
        in_deg_local = {x:0 for x in comp}
        out_deg_local = {x:0 for x in comp}
        comp_set = set(comp)
        for u in comp:
            for v in scc_graph[u]:
                if v in comp_set:
                    out_deg_local[u] += 1
                    in_deg_local[v] += 1

        # Now count how many are sources/sinks locally
        count_source = 0
        count_sink = 0
        for u in comp:
            if in_deg_local[u] == 0: 
                count_source += 1
            if out_deg_local[u] == 0:
                count_sink += 1

        if len(comp) == 1:
            # If there's a single SCC node with no edges to anything else in this subcomponent,
            # then it is both a source and a sink => #sources=1, #sinks=1
            pass
        # The number of strings needed for this subcomponent is max(count_source, count_sink).
        # But note if the subcomponent has no edges at all (which means it wasn't actually used),
        # it wouldn't appear in the Tarjan decomposition with an SCC_id we care about.
        # So we assume there's at least one used edge or something in that SCC.
        # There's a corner case: an SCC with a single vertex that really does not appear in any used edge?
        # Actually we took care only to run Tarjan if that vertex had in or out used edges.

        # There's a known corner case: if the subcomponent is nonempty but has no edges,
        # that would mean there's a single vertex with no self-loop. Then sources=1, sinks=1 => contributes 1.
        # That is consistent with the formula.

        # Finally compute the needed paths:
        needed = max(count_source, count_sink)
        if needed == 0:
            # This can happen if everything is in one subcomponent and it has cycles.
            # Typically that means the entire subcomponent is strongly connected and
            # there's at least one vertex or edge. Then effectively it's one big cycle,
            # so #sources=0, #sinks=0. But we do need at least 1 path to cover it.
            needed = 1
        answer += needed

    print(answer)