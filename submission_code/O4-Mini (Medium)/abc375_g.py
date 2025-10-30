import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N, M = map(int, input().split())
    A = [0] * M
    B = [0] * M
    C = [0] * M
    adjG = [[] for _ in range(N+1)]
    for i in range(M):
        a,b,c = map(int, input().split())
        A[i], B[i], C[i] = a, b, c
        adjG[a].append((b, c))
        adjG[b].append((a, c))

    import heapq
    INF = 10**30

    # Dijkstra from 1
    dist1 = [INF] * (N+1)
    dist1[1] = 0
    hq = [(0, 1)]
    while hq:
        d, u = heapq.heappop(hq)
        if d > dist1[u]:
            continue
        for v, cost in adjG[u]:
            nd = d + cost
            if nd < dist1[v]:
                dist1[v] = nd
                heapq.heappush(hq, (nd, v))

    # Dijkstra from N
    distN = [INF] * (N+1)
    distN[N] = 0
    hq = [(0, N)]
    while hq:
        d, u = heapq.heappop(hq)
        if d > distN[u]:
            continue
        for v, cost in adjG[u]:
            nd = d + cost
            if nd < distN[v]:
                distN[v] = nd
                heapq.heappush(hq, (nd, v))

    D = dist1[N]

    # Build shortest-path DAG edges (directed), then undirected for bridge finding
    dag_edge_id = [-1] * M
    dag_edges = []  # list of (u, v, original_index)
    for i in range(M):
        u, v, c = A[i], B[i], C[i]
        # orientation u->v?
        if dist1[u] + c + distN[v] == D:
            dag_edge_id[i] = len(dag_edges)
            dag_edges.append((u, v, i))
        elif dist1[v] + c + distN[u] == D:
            dag_edge_id[i] = len(dag_edges)
            dag_edges.append((v, u, i))
        # else not in DAG

    K = len(dag_edges)
    # build undirected adjacency for the DAG edges
    adj = [[] for _ in range(N+1)]
    for eid, (u, v, _) in enumerate(dag_edges):
        adj[u].append((v, eid))
        adj[v].append((u, eid))

    # Bridge finding on undirected graph adj, only component reachable from 1
    visited = [False] * (N+1)
    tin = [0] * (N+1)
    low = [0] * (N+1)
    is_bridge = [False] * K
    timer = 0

    # iterative DFS stack: (node, parent_node, parent_eid, next_child_index)
    stack = [(1, -1, -1, 0)]
    while stack:
        v, p, p_eid, idx = stack.pop()
        if idx == 0:
            visited[v] = True
            tin[v] = timer
            low[v] = timer
            timer += 1
        if idx < len(adj[v]):
            to, eid = adj[v][idx]
            # push current frame with next index
            stack.append((v, p, p_eid, idx+1))
            if eid == p_eid:
                # don't go back via parent edge
                continue
            if visited[to]:
                # back-edge or cross-edge
                if tin[to] < low[v]:
                    low[v] = tin[to]
                continue
            else:
                # tree-edge, go deeper
                stack.append((to, v, eid, 0))
                continue
        else:
            # finished all children of v
            if p != -1:
                # check if (p -- v) is a bridge
                if low[v] > tin[p]:
                    # edge p_eid is a bridge
                    is_bridge[p_eid] = True
                # propagate low to parent
                if low[v] < low[p]:
                    low[p] = low[v]
            # no explicit return, frame ends

    out = []
    for i in range(M):
        eid = dag_edge_id[i]
        if eid != -1 and is_bridge[eid]:
            out.append("Yes")
        else:
            out.append("No")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()