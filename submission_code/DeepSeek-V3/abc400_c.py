import sys
from collections import deque

def solve():
    N, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
    
    color = [None] * (N + 1)
    total_possible = 0
    visited = [False] * (N + 1)
    
    for i in range(1, N+1):
        if not visited[i]:
            queue = deque()
            queue.append(i)
            color[i] = 0
            cnt = [0, 0]
            cnt[0] += 1
            visited[i] = True
            is_bipartite = True
            while queue:
                v = queue.popleft()
                for neighbor in adj[v]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        color[neighbor] = color[v] ^ 1
                        cnt[color[neighbor]] += 1
                        queue.append(neighbor)
                    else:
                        if color[neighbor] == color[v]:
                            is_bipartite = False
            if not is_bipartite:
                # According to problem statement, initial graph is bipartite, so this shouldn't happen
                pass
            total_possible += cnt[0] * cnt[1]
    
    remaining = total_possible - M
    if remaining % 2 == 1:
        print("Aoki")
    else:
        print("Takahashi")

solve()