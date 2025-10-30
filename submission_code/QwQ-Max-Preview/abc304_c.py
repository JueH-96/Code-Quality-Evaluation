import sys
from collections import deque

def main():
    n, d = map(int, sys.stdin.readline().split())
    points = [tuple(map(int, sys.stdin.readline().split())) for _ in range(n)]
    
    d_sq = d * d
    adj = [[] for _ in range(n)]
    
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
            if dist_sq <= d_sq:
                adj[i].append(j)
                adj[j].append(i)
    
    visited = [False] * n
    q = deque()
    visited[0] = True
    q.append(0)
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                q.append(v)
    
    for ans in visited:
        print("Yes" if ans else "No")

if __name__ == "__main__":
    main()