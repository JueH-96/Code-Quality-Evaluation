def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data=sys.stdin.read().strip().split()
    N,M= map(int,input_data[:2])
    edges = []
    idx = 2
    for i in range(M):
        A = int(input_data[idx]); B = int(input_data[idx+1]); C = int(input_data[idx+2])
        edges.append((A,B,C))
        idx += 3

    #--------------------------------------------------------------------------
    # Step 1: Compute shortest distances D1(x) from 1 to x
    #         and DN(x) from x to N using Dijkstra's algorithm.
    #--------------------------------------------------------------------------

    import heapq

    # Build adjacency for the full graph to run Dijkstra
    adj_full = [[] for _ in range(N+1)]
    for i,(a,b,c) in enumerate(edges):
        adj_full[a].append((b,c))
        adj_full[b].append((a,c))

    INF = 10**18

    def dijkstra(start):
        dist = [INF]*(N+1)
        dist[start] = 0
        pq = [(0,start)]
        while pq:
            cd,u = heapq.heappop(pq)
            if dist[u]<cd:
                continue
            for v, w in adj_full[u]:
                nd = cd + w
                if nd<dist[v]:
                    dist[v]=nd
                    heapq.heappush(pq,(nd,v))
        return dist

    D1 = dijkstra(1)
    DN = dijkstra(N)

    d0 = D1[N]  # the shortest distance from 1 to N using all roads
    # Per problem statement, d0 is guaranteed finite ("City N can be reached when all roads are passable").

    #--------------------------------------------------------------------------
    # Step 2: Build the subgraph G' of edges that lie on at least one shortest path
    #         from 1 to N. For an edge (u,v,c), it is in G' if
    #         D1[u] + c + DN[v] == d0 or D1[v] + c + DN[u] == d0.
    #--------------------------------------------------------------------------

    # We'll keep an adjacency list for G' and also remember which edges are included.
    in_shortest_graph = [False]*M
    adj_gprime = [[] for _ in range(N+1)]
    for i,(a,b,c) in enumerate(edges):
        if D1[a] + c + DN[b] == d0 or D1[b] + c + DN[a] == d0:
            # This edge is in G'
            in_shortest_graph[i] = True
            adj_gprime[a].append((b,i))
            adj_gprime[b].append((a,i))

    #--------------------------------------------------------------------------
    # Step 3: Find all "bridges" in G' using a standard DFS-based algorithm.
    #         We only care about edges in G'.
    #
    # An edge e=(u,v) in G' is a "bridge in G'" if removing it increases the number of
    # connected components of G'.
    #
    # We'll store them and then figure out which ones actually separate 1 from N.
    #--------------------------------------------------------------------------

    sys.setrecursionlimit(10**7)
    used = [False]*(N+1)
    tin = [0]*(N+1)
    low = [0]*(N+1)
    timer = 0

    bridges = []  # will store (u,v,idx) for each bridge-edge in G'
    # We'll do a standard "bridges in undirected graph" DFS.

    def dfs_bridge(u, p_edge_idx=-1):
        nonlocal timer
        used[u] = True
        timer+=1
        tin[u]=low[u]=timer
        for v,eid in adj_gprime[u]:
            if eid == p_edge_idx:
                # skip the edge to parent in DFS tree
                continue
            if not used[v]:
                dfs_bridge(v, eid)
                low[u] = min(low[u], low[v])
                # check if it's a bridge
                if low[v] > tin[u]:
                    # the edge (u,v,eid) is a bridge
                    bridges.append((u,v,eid))
            else:
                # back edge
                low[u] = min(low[u], tin[v])

    for v in range(1, N+1):
        if not used[v]:
            dfs_bridge(v,-1)

    #--------------------------------------------------------------------------
    # Step 4: We now know which edges in G' are bridges. However, an edge can be
    #         a bridge but might not necessarily separate 1 from N. We only
    #         answer "Yes" for an edge in G' if it is indeed a bridge that,
    #         upon removal, puts 1 and N in different connected components.
    #
    #         A well-known approach is to form the "2-edge-connected components"
    #         after removing these bridges, label them, then build a "bridge tree."
    #         The path in that tree between comp(1) and comp(N) is unique. Any
    #         bridge on that path will separate 1 and N. Others won't.
    #
    #         We will:
    #         - Remove (ignore) all bridges from G', find connected components
    #           of the remaining edges. That gives us a component ID for each vertex.
    #         - Build the "bridge tree" where each bridge connects two of these components.
    #         - Do a BFS from comp(1) to comp(N) in that tree. The edges we use
    #           in that BFS are exactly the bridges that separate 1 and N.
    #--------------------------------------------------------------------------

    # Mark each edge if it is a bridge in G'
    is_bridge_in_gprime = [False]*M
    for u,v,eid in bridges:
        is_bridge_in_gprime[eid] = True

    # Build adjacency of "non-bridge" edges in G'
    adj_nobridge = [[] for _ in range(N+1)]
    for i,(a,b,c) in enumerate(edges):
        if in_shortest_graph[i] and (not is_bridge_in_gprime[i]):
            # This edge is in G' but not a bridge
            adj_nobridge[a].append(b)
            adj_nobridge[b].append(a)

    # Find components ignoring all bridges
    comp = [-1]*(N+1)
    comp_id = 0

    def dfs_comp(s):
        stack = [s]
        comp[s] = comp_id
        top = 0
        while stack:
            u = stack.pop()
            for w in adj_nobridge[u]:
                if comp[w] < 0:
                    comp[w] = comp_id
                    stack.append(w)

    for v in range(1,N+1):
        if comp[v]<0:
            dfs_comp(v)
            comp_id+=1

    # Now build the "bridge tree." Each of the comp_id components is a node in this tree.
    # Each bridge in G' is an edge connecting comp[u] and comp[v].
    # We'll keep track of the original edge index as well.
    bridge_tree = [[] for _ in range(comp_id)]
    for u,v,eid in bridges:
        cu = comp[u]
        cv = comp[v]
        if cu!=cv:
            bridge_tree[cu].append((cv,eid))
            bridge_tree[cv].append((cu,eid))

    # We'll do a BFS from comp(1) to comp(N) in the bridge tree, mark
    # which bridge edges are used on that unique path. The edges on that path
    # are exactly those that separate 1 and N.
    start_comp = comp[1]
    end_comp = comp[N]

    from collections import deque
    distBT = [-1]*comp_id
    parent_edge = [-1]*comp_id  # store the edge idx by which we come
    distBT[start_comp] = 0
    queue = deque([start_comp])
    while queue:
        cc = queue.popleft()
        if cc == end_comp:
            break
        for nxt,eid in bridge_tree[cc]:
            if distBT[nxt]<0:
                distBT[nxt] = distBT[cc]+1
                parent_edge[nxt] = eid
                queue.append(nxt)

    used_bridge_on_path = set()
    if distBT[end_comp]>=0:
        # Reconstruct path in the bridge tree
        cur = end_comp
        while cur!=start_comp:
            e_id = parent_edge[cur]
            used_bridge_on_path.add(e_id)
            # move to the component on the other side of that bridge
            # find the other end's component
            u,v = edges[e_id][0], edges[e_id][1]
            cU = comp[u]; cV = comp[v]
            if cU == cur:
                cur = cV
            else:
                cur = cU

    #--------------------------------------------------------------------------
    # Step 5: Produce the final answer:
    #   - If an edge is not in G', answer "No" (it was never on a shortest path).
    #   - If an edge is in G' but not a bridge, answer "No" (removing it doesn't change the short dist).
    #   - If an edge is in G' and is a bridge but not on the 1--N path in the "bridge tree", answer "No".
    #   - If an edge is in G' and is a bridge and is on that path, answer "Yes".
    #
    # Also note the spec: if removing that edge makes 1->N unreachable while it was reachable
    # or vice versa, that also counts as "Yes". Exactly the same logic: only those
    # edges that are "critical" in G' for connecting 1 to N are "Yes".
    #--------------------------------------------------------------------------

    ans = []
    for i in range(M):
        if not in_shortest_graph[i]:
            # This edge is not on any shortest path => removing it doesn't affect shortest path length
            ans.append("No")
        else:
            # It's in G'
            if not is_bridge_in_gprime[i]:
                ans.append("No")
            else:
                # It's a bridge in G'
                if i in used_bridge_on_path:
                    ans.append("Yes")
                else:
                    ans.append("No")

    print('
'.join(ans))

# Don't forget to call main()
if __name__ == "__main__":
    main()