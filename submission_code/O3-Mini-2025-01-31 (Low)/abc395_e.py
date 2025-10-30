def main():
    import sys, heapq

    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    X = int(next(it))

    # Build two graphs.
    # graph_norm[u] contains vertices v such that there is an edge u -> v (used in state 0)
    # graph_rev[u] contains vertices v such that there is an edge u -> v in the reversed graph,
    # which originally corresponds to an edge v -> u (used in state 1)
    graph_norm = [[] for _ in range(n + 1)]
    graph_rev = [[] for _ in range(n + 1)]
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        graph_norm[u].append(v)
        graph_rev[v].append(u)

    # We use Dijkstra's algorithm on a state space where a state is (vertex, parity).
    # Parity 0 means edges are in the original direction,
    # Parity 1 means edges are reversed.
    # Transitions:
    #  - If in state (v, par), moving along an edge has cost 1.
    #    * If par == 0, follow an edge from v in graph_norm.
    #    * If par == 1, follow an edge from v in graph_rev.
    #  - It is also allowed to reverse all edges at vertex v.
    #    This changes the state from (v, par) to (v, 1-par) with cost X.
    INF = 10**20
    # distance[v][par] is the best cost to reach vertex v with state par.
    dist = [[INF, INF] for _ in range(n + 1)]
    dist[1][0] = 0

    heap = [(0, 1, 0)]  # (cost, vertex, parity)
    while heap:
        cost, u, par = heapq.heappop(heap)
        if cost != dist[u][par]:
            continue

        # Try to move along directed edges.
        if par == 0:
            for v in graph_norm[u]:
                nc = cost + 1
                if nc < dist[v][par]:
                    dist[v][par] = nc
                    heapq.heappush(heap, (nc, v, par))
        else:
            for v in graph_rev[u]:
                nc = cost + 1
                if nc < dist[v][par]:
                    dist[v][par] = nc
                    heapq.heappush(heap, (nc, v, par))

        # Try to reverse the direction of all edges (toggle state).
        npc = cost + X
        npar = 1 - par
        if npc < dist[u][npar]:
            dist[u][npar] = npc
            heapq.heappush(heap, (npc, u, npar))

    ans = min(dist[n][0], dist[n][1])
    sys.stdout.write(str(ans))


if __name__ == '__main__':
    main()