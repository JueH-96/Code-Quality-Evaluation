def main():
    import sys, sys
    import heapq
    data = sys.stdin.buffer.read().split()
    # parse input
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    S = int(next(it)) - 1
    T = int(next(it)) - 1
    
    # We will build a flow network using vertex splitting.
    # Special vertices: S and T will not be split.
    # All other vertices v get two nodes: v_in and v_out.
    #
    # We'll assign new indices as follows:
    # For vertex v:
    #   if v is S or T, use one node. Otherwise assign two nodes.
    #
    # Let new_index count.
    # We also want functions node_in(v) and node_out(v).
    node_in = [None] * n
    node_out = [None] * n
    new_index = 0
    for v in range(n):
        if v == S or v == T:
            node_in[v] = new_index
            node_out[v] = new_index
            new_index += 1
        else:
            node_in[v] = new_index
            new_index += 1
            node_out[v] = new_index
            new_index += 1

    N_flow = new_index  # total nodes in flow graph

    # Build the graph structure for mincost flow.
    # We represent each edge with a tuple: [to, cap, cost, rev]
    graph = [[] for _ in range(N_flow)]
    
    def add_edge(frm, to, cap, cost):
        graph[frm].append([to, cap, cost, len(graph[to])])
        graph[to].append([frm, 0, -cost, len(graph[frm]) - 1])
    
    # For every vertex v (which is not special), add edge from v_in to v_out (cap=1, cost=0)
    for v in range(n):
        if v == S or v == T: continue
        add_edge(node_in[v], node_out[v], 1, 0)
    
    # Process each original undirected edge.
    # For an undirected edge (u,v) add (u_out -> v_in) and (v_out -> u_in) with capacity=1, cost=1.
    for _ in range(m):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        # For special vertices (S or T) we use the same index for in & out.
        add_edge(node_out[u], node_in[v], 1, 1)
        add_edge(node_out[v], node_in[u], 1, 1)
    
    # Now, we run min cost flow from S_flow to T_flow (they are node_in[S] and node_in[T] since S and T are not split).
    source = node_in[S]
    sink = node_in[T]
    flow_limit = 2  # we need 2 vertex-disjoint S-T paths
    
    INF = 10**9
    res = 0
    # potential for reduced costs (initially 0)
    potential = [0] * N_flow

    # mincost flow (Dijkstra based) for sending f units flow
    f = flow_limit
    while f:
        dist = [INF] * N_flow
        dist[source] = 0
        prev_v = [-1] * N_flow
        prev_e = [-1] * N_flow
        hq = [(0, source)]
        while hq:
            d, v = heapq.heappop(hq)
            if dist[v] != d: continue
            for i, (to, cap, cost, rev) in enumerate(graph[v]):
                if cap > 0 and dist[to] > d + cost + potential[v] - potential[to]:
                    dist[to] = d + cost + potential[v] - potential[to]
                    prev_v[to] = v
                    prev_e[to] = i
                    heapq.heappush(hq, (dist[to], to))
        if dist[sink] == INF:
            print(-1)
            return
        for v in range(N_flow):
            if dist[v] < INF:
                potential[v] += dist[v]
        # How much flow can we add (should be 1 because of capacities=1 along splitting edges)
        add_flow = f
        v = sink
        while v != source:
            pv = prev_v[v]
            pe = prev_e[v]
            add_flow = min(add_flow, graph[pv][pe][1])
            v = pv
        f -= add_flow
        res += add_flow * potential[sink]
        v = sink
        while v != source:
            pv = prev_v[v]
            pe = prev_e[v]
            graph[pv][pe][1] -= add_flow
            rev = graph[pv][pe][3]
            graph[v][rev][1] += add_flow
            v = pv

    # The cost computed equals the sum of the lengths of the two disjoint S-T paths.
    # That is exactly the minimum number of moves.
    print(res)

if __name__ == '__main__':
    main()