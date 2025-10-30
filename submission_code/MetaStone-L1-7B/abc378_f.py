import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N+1)]
    edges = []
    for _ in range(N-1):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
        edges.append((u, v))
    
    degrees = [0] * (N + 1)
    for u in range(1, N+1):
        degrees[u] = len(adj[u])
    
    leaves = [u for u in range(1, N+1) if degrees[u] == 1]
    
    visited = [False] * (N + 1)
    count = 0
    
    for u in leaves:
        if not visited[u]:
            queue = deque()
            queue.append(u)
            visited[u] = True
            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if degrees[neighbor] == 2 and not visited[neighbor]:
                        if degrees[neighbor] == 1:
                            count += 1
                            visited[neighbor] = True
                        else:
                            visited[neighbor] = True
                            queue.append(neighbor)
    
    print(count // 2)

if __name__ == "__main__":
    main()