import sys
import heapq

def dijkstra(n, graph, start):
    d = [float('inf')] * (n + 1)
    d[start] = 0
    h = [(0, start)]
    while h:
        dist, u = heapq.heappop(h)
        if dist > d[u]:
            continue
        for v, c in graph[u]:
            if d[v] > d[u] + c:
                d[v] = d[u] + c
                heapq.heappush(h, (d[v], v))
    return d

def main():
    import sys
    from collections import defaultdict, deque

    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    edges = []
    graph = [[] for _ in range(N + 1)]
    for _ in range(M):
        A = int(data[idx])
        idx += 1
        B = int(data[idx])
        idx += 1
        C = int(data[idx])
        idx += 1
        edges.append((A, B, C))
        graph[A].append((B, C))
        graph[B].append((A, C))
    
    d_all = dijkstra(N, graph, 1)
    d_rev_graph = [[] for _ in range(N + 1)]
    for A, B, C in edges:
        d_rev_graph[B].append((A, C))
        d_rev_graph[A].append((B, C))
    d_rev = dijkstra(N, d_rev_graph, N)
    
    subgraph = [[] for _ in range(N + 1)]
    edge_list = []
    for i, (A, B, C) in enumerate(edges):
        if d_all[A] + C + d_rev[B] == d_all[N]:
            subgraph[A].append(B)
            edge_list.append((A, B, i))
        if d_all[B] + C + d_rev[A] == d_all[N]:
            subgraph[B].append(A)
            edge_list.append((B, A, i))
    
    levels = [0] * (N + 1)
    for u in range(1, N + 1):
        levels[u] = d_all[u]
    nodes = sorted(range(1, N + 1), key=lambda x: levels[x])
    
    P = [0] * (N + 1)
    P[1] = 1
    for u in nodes:
        for v in subgraph[u]:
            P[v] += P[u]
    
    Q = [0] * (N + 1)
    Q[N] = 1
    for u in reversed(nodes):
        for v in subgraph[u]:
            Q[u] += Q[v]
    
    is_critical = [False] * M
    edge_to_index = [set() for _ in range(M)]
    for u, v, i in edge_list:
        edge_to_index[i].add((u, v))
    
    for u, v, i in edge_list:
        if P[u] * Q[v] == P[v]:
            is_critical[i] = True
    
    for i in range(M):
        if is_critical[i]:
            print("Yes")
        else:
            print("No")

if __name__ == '__main__':
    main()