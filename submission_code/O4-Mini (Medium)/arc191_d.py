import sys
import threading
def main():
    import sys
    import heapq

    sys.setrecursionlimit(1 << 25)
    data = sys.stdin

    # Read input
    line = data.readline().split()
    N = int(line[0])
    M = int(line[1])
    S = int(line[2])
    T = int(line[3])

    edges = [tuple(map(int, data.readline().split())) for _ in range(M)]

    # We will build a flow graph with vertex-splitting for all vertices != S,T.
    # For v == S or v == T, we do NOT split; we treat in/out as the same node.
    # Map each vertex to node-ids in the flow graph:
    #   If v == S or T: id_in[v] = id_out[v] = one new id
    #   Else:               id_in[v], id_out[v] = two consecutive new ids
    id_in = [0] * (N + 1)
    id_out = [0] * (N + 1)
    nid = 0
    for v in range(1, N+1):
        if v == S or v == T:
            id_in[v] = nid
            id_out[v] = nid
            nid += 1
        else:
            id_in[v] = nid
            nid += 1
            id_out[v] = nid
            nid += 1
    # total number of nodes
    n_nodes = nid

    # adjacency list for min-cost max-flow
    # each edge: (to, rev_index, cap, cost)
    G = [[] for _ in range(n_nodes)]

    def add_edge(u, v, cap, cost):
        # forward edge index = len(G[u])
        # backward edge index = len(G[v])
        G[u].append([v, len(G[v]), cap, cost])
        G[v].append([u, len(G[u]) - 1, 0, -cost])

    # For each vertex v != S,T, add in->out with cap=1 cost=0
    for v in range(1, N+1):
        if v != S and v != T:
            add_edge(id_in[v], id_out[v], 1, 0)

    # For each undirected edge u-v, add two directed edges in the flow network:
    # u_out -> v_in  cap=1 cost=1
    # v_out -> u_in  cap=1 cost=1
    for u, v in edges:
        uo = id_out[u]
        vi = id_in[v]
        add_edge(uo, vi, 1, 1)
        vo = id_out[v]
        ui = id_in[u]
        add_edge(vo, ui, 1, 1)

    # We'll send flow = 2 from source = id_out[S] to sink = id_in[T],
    # and compute the minimum cost.  If we cannot send 2 units, answer = -1.
    source = id_out[S]
    sink = id_in[T]
    flow_needed = 2

    INF = 10**30
    n = n_nodes
    potential = [0] * n   # potentials for reduced costs
    dist = [0] * n        # distances in Dijkstra
    prevv = [0] * n       # previous vertex on the augmenting path
    preve = [0] * n       # which edge index we used

    flow = 0
    cost = 0

    while flow < flow_needed:
        # Dijkstra to find shortest augmenting path (in reduced costs)
        for i in range(n):
            dist[i] = INF
        dist[source] = 0
        hq = [(0, source)]
        while hq:
            d, v = heapq.heappop(hq)
            if d > dist[v]:
                continue
            for ei, e in enumerate(G[v]):
                if e[2] > 0:
                    # e = [to, rev, cap, cost]
                    to = e[0]
                    c_cost = e[3]
                    nd = d + c_cost + potential[v] - potential[to]
                    if nd < dist[to]:
                        dist[to] = nd
                        prevv[to] = v
                        preve[to] = ei
                        heapq.heappush(hq, (nd, to))
        if dist[sink] == INF:
            # cannot send more flow
            break
        # update potentials
        for v in range(n):
            if dist[v] < INF:
                potential[v] += dist[v]
        # augment by 1 unit
        addf = 1
        flow += addf
        # the cost of this unit is the potential of sink (since pot[source] stays 0)
        cost += potential[sink] * addf
        # walk back and update capacities
        v = sink
        while v != source:
            pv = prevv[v]
            pe = preve[v]
            G[pv][pe][2] -= addf      # reduce cap on forward edge
            rev = G[pv][pe][1]
            G[v][rev][2] += addf      # increase cap on reverse edge
            v = pv

    # Check if we got required flow
    if flow < flow_needed:
        print(-1)
    else:
        print(cost)

if __name__ == "__main__":
    main()