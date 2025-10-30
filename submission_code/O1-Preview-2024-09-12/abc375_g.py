# YOUR CODE HERE
import sys
import threading
import heapq
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    input = sys.stdin.readline

    N, M = map(int, sys.stdin.readline().split())
    edges = []
    graph_dijkstra = [[] for _ in range(N)]
    graph_tarjan = [[] for _ in range(N)]

    for idx in range(M):
        A_i, B_i, C_i = map(int, sys.stdin.readline().split())
        A_i -= 1  # zero-based indexing
        B_i -= 1
        edges.append((A_i, B_i, C_i, idx))  # (u, v, cost, index)
        # Build adjacency list for Dijkstra
        graph_dijkstra[A_i].append((B_i, C_i))
        graph_dijkstra[B_i].append((A_i, C_i))
        # Build adjacency list for Tarjan (edge index needed)
        graph_tarjan[A_i].append((B_i, idx))
        graph_tarjan[B_i].append((A_i, idx))

    # Dijkstra from node 1
    import heapq

    dist1 = [float('inf')] * N
    dist1[0] = 0
    hq = [(0, 0)]  # (distance, node)
    while hq:
        d, u = heapq.heappop(hq)
        if dist1[u] < d:
            continue
        for v, w in graph_dijkstra[u]:
            if dist1[v] > dist1[u] + w:
                dist1[v] = dist1[u] + w
                heapq.heappush(hq, (dist1[v], v))

    # Dijkstra from node N
    distN = [float('inf')] * N
    distN[N - 1] = 0
    hq = [(0, N - 1)]  # (distance, node)
    while hq:
        d, u = heapq.heappop(hq)
        if distN[u] < d:
            continue
        for v, w in graph_dijkstra[u]:
            if distN[v] > distN[u] + w:
                distN[v] = distN[u] + w
                heapq.heappush(hq, (distN[v], v))

    dist_to_N = dist1[N - 1]

    is_on_shortest_path = [False] * M

    for u, v, c, idx in edges:
        # Check if edge is on any shortest path
        # Since edges are bidirectional, check both directions
        if dist1[u] + c + distN[v] == dist_to_N:
            is_on_shortest_path[idx] = True
        elif dist1[v] + c + distN[u] == dist_to_N:
            is_on_shortest_path[idx] = True

    # Tarjan's Algorithm to find bridges
    time = 0
    tin = [-1] * N
    low = [-1] * N
    visited = [False] * N
    is_bridge = [False] * M

    def dfs(u, parent_edge_idx):
        nonlocal time
        visited[u] = True
        tin[u] = time
        low[u] = time
        time += 1
        for v, edge_idx in graph_tarjan[u]:
            if edge_idx == parent_edge_idx:
                continue
            if visited[v]:
                low[u] = min(low[u], tin[v])
            else:
                dfs(v, edge_idx)
                low[u] = min(low[u], low[v])
                if low[v] > tin[u]:
                    # Edge (u,v) is a bridge
                    is_bridge[edge_idx] = True

    dfs(0, -1)

    # For each edge, output 'Yes' or 'No'
    for idx in range(M):
        if is_on_shortest_path[idx] or is_bridge[idx]:
            print('Yes')
        else:
            print('No')

threading.Thread(target=main).start()