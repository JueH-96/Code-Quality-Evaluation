import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    
    adj = [[] for _ in range(N+1)]
    edges_to_1 = []
    
    for _ in range(M):
        a = int(input[idx])
        idx += 1
        b = int(input[idx])
        idx += 1
        adj[a].append(b)
        if b == 1:
            edges_to_1.append(a)
    
    INF = float('inf')
    dist = [INF] * (N + 1)
    dist[1] = 0
    q = deque()
    q.append(1)
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == INF:
                dist[v] = dist[u] + 1
                q.append(v)
    
    min_cycle = INF
    for u in edges_to_1:
        if dist[u] != INF:
            if dist[u] + 1 < min_cycle:
                min_cycle = dist[u] + 1
    
    if min_cycle == INF:
        print(-1)
    else:
        print(min_cycle)

if __name__ == "__main__":
    main()