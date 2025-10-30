import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N+1)]
    reversed_adj = [[] for _ in range(N+1)]
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        adj[a].append(b)
        reversed_adj[b].append(a)
    
    # Compute forward distances (d1) from 1
    INF = float('inf')
    d1 = [INF] * (N + 1)
    d1[1] = 0
    q = deque([1])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if d1[v] == INF:
                d1[v] = d1[u] + 1
                q.append(v)
    
    # Compute reverse distances (d_rev) from 1 in reversed graph (distance to 1 in original)
    d_rev = [INF] * (N + 1)
    d_rev[1] = 0
    q = deque([1])
    while q:
        u = q.popleft()
        for v in reversed_adj[u]:
            if d_rev[v] == INF:
                d_rev[v] = d_rev[u] + 1
                q.append(v)
    
    min_cycle = INF
    for u in range(1, N+1):
        if u == 1:
            continue
        if d1[u] != INF and d_rev[u] != INF:
            total = d1[u] + d_rev[u]
            if total < min_cycle:
                min_cycle = total
    
    print(min_cycle if min_cycle != INF else -1)

if __name__ == "__main__":
    main()