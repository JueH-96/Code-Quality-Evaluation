import sys
from collections import deque

def main():
    n, k = map(int, sys.stdin.readline().split())
    edges = [tuple(map(int, sys.stdin.readline().split())) for _ in range(n-1)]
    terminals = set(map(int, sys.stdin.readline().split()))
    
    adj = [[] for _ in range(n + 1)]
    degree = [0] * (n + 1)
    
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
        degree[a] += 1
        degree[b] += 1
    
    removed = [False] * (n + 1)
    q = deque()
    
    for node in range(1, n + 1):
        if node not in terminals and degree[node] == 1:
            q.append(node)
    
    while q:
        u = q.popleft()
        if removed[u]:
            continue
        removed[u] = True
        for v in adj[u]:
            if not removed[v]:
                degree[v] -= 1
                if v not in terminals and degree[v] == 1:
                    q.append(v)
    
    ans = sum(1 for node in range(1, n + 1) if not removed[node])
    print(ans)

if __name__ == "__main__":
    main()