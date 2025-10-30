import sys
import threading
def main():
    import sys
    input = sys.stdin.readline
    N, M = map(int, input().split())
    # Read edges and build undirected graph for Dijkstra
    edges = [None] * M
    graph = [[] for _ in range(N+1)]
    for i in range(M):
        u, v, w = map(int, input().split())
        edges[i] = (u, v, w)
        graph[u].append((v, w))
        graph[v].append((u, w))
    # Dijkstra from start node
    import heapq
    INF = 10**30
    def dijkstra(start):
        dist = [INF] * (N+1)
        dist[start] = 0
        hq = [(0, start)]
        while hq:
            d, u = heapq.heappop(hq)
            if d != dist[u]:
                continue
            for v, w in graph[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(hq, (nd, v))
        return dist
    d1 = dijkstra(1)
    dN = dijkstra(N)
    D = d1[N]
    # Identify nodes on any shortest path
    spnode = [False] * (N+1)
    for u in range(1, N+1):
        if d1[u] + dN[u] == D:
            spnode[u] = True
    # Build the shortest-path DAG (directed)
    sp_adj = [[] for _ in range(N+1)]
    sp_rev = [[] for _ in range(N+1)]
    for u, v, w in edges:
        # orientation u->v?
        if spnode[u] and spnode[v] and d1[u] + w == d1[v]:
            sp_adj[u].append(v)
            sp_rev[v].append(u)
        # orientation v->u?
        elif spnode[u] and spnode[v] and d1[v] + w == d1[u]:
            sp_adj[v].append(u)
            sp_rev[u].append(v)
    # Collect SP-nodes in topological order by d1
    sp_nodes = [u for u in range(1, N+1) if spnode[u]]
    sp_nodes.sort(key=lambda x: d1[x])
    # Count number of shortest paths from 1 to each node (mod two primes)
    mod1 = 10**9 + 7
    mod2 = 10**9 + 9
    f1 = [0] * (N+1)
    f2 = [0] * (N+1)
    f1[1] = 1
    f2[1] = 1
    for u in sp_nodes:
        fu1 = f1[u]
        fu2 = f2[u]
        if fu1 or fu2:
            for v in sp_adj[u]:
                f1[v] = (f1[v] + fu1) % mod1
                f2[v] = (f2[v] + fu2) % mod2
    total1 = f1[N]
    total2 = f2[N]
    # Count number of shortest paths from each node to N
    g1 = [0] * (N+1)
    g2 = [0] * (N+1)
    g1[N] = 1
    g2[N] = 1
    for u in reversed(sp_nodes):
        gu1 = g1[u]
        gu2 = g2[u]
        if gu1 or gu2:
            for v in sp_rev[u]:
                g1[v] = (g1[v] + gu1) % mod1
                g2[v] = (g2[v] + gu2) % mod2
    # Determine for each edge if it is critical
    out = []
    for u, v, w in edges:
        critical = False
        # check orientation u->v
        if d1[u] + w + dN[v] == D:
            # if all shortest paths go through this directed edge
            if (f1[u] * g1[v]) % mod1 == total1 and (f2[u] * g2[v]) % mod2 == total2:
                critical = True
        # check orientation v->u
        elif d1[v] + w + dN[u] == D:
            if (f1[v] * g1[u]) % mod1 == total1 and (f2[v] * g2[u]) % mod2 == total2:
                critical = True
        # else not on any shortest path => not critical
        out.append("Yes" if critical else "No")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()