import sys
import heapq
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    edges = []
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        a, b, c = map(int, sys.stdin.readline().split())
        edges.append((a, b, c))
        adj[a].append((b, c))
        adj[b].append((a, c))
    
    # Compute d_start (distance from 1)
    def dijkstra(start, adj, n):
        INF = float('inf')
        d = [INF] * (n + 1)
        d[start] = 0
        heap = [(0, start)]
        while heap:
            dist_u, u = heapq.heappop(heap)
            if dist_u > d[u]:
                continue
            for v, c in adj[u]:
                if d[v] > d[u] + c:
                    d[v] = d[u] + c
                    heapq.heappush(heap, (d[v], v))
        return d
    
    d_start = dijkstra(1, adj, N)
    d_end = dijkstra(N, adj, N)
    D = d_start[N]
    
    # Build DAG
    dag_adj = [[] for _ in range(N+1)]
    for a, b, c in edges:
        if d_start[a] + c + d_end[b] == D:
            dag_adj[a].append(b)
        if d_start[b] + c + d_end[a] == D:
            dag_adj[b].append(a)
    
    # Compute topological order using Kahn's algorithm
    in_degree = [0] * (N+1)
    for u in range(1, N+1):
        for v in dag_adj[u]:
            in_degree[v] += 1
    
    q = deque()
    for u in range(1, N+1):
        if in_degree[u] == 0:
            q.append(u)
    
    top_order = []
    while q:
        u = q.popleft()
        top_order.append(u)
        for v in dag_adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    
    # Compute in_count
    in_count = [0] * (N+1)
    in_count[1] = 1
    for u in top_order:
        for v in dag_adj[u]:
            in_count[v] += in_count[u]
    
    # Compute out_count
    out_count = [0] * (N+1)
    out_count[N] = 1
    reverse_order = top_order[::-1]
    for v in reverse_order:
        for w in dag_adj[v]:
            out_count[v] += out_count[w]
    
    total_paths = in_count[N]
    
    # Process each edge
    for a, b, c in edges:
        cond1 = (d_start[a] + c + d_end[b] == D)
        cond2 = (d_start[b] + c + d_end[a] == D)
        if not (cond1 or cond2):
            print("No")
        else:
            if cond1 and cond2:
                print("No")
            elif cond1:
                u, v = a, b
                if in_count[u] * out_count[v] == total_paths:
                    print("Yes")
                else:
                    print("No")
            elif cond2:
                u, v = b, a
                if in_count[u] * out_count[v] == total_paths:
                    print("Yes")
                else:
                    print("No")
            else:
                print("No")

if __name__ == "__main__":
    main()