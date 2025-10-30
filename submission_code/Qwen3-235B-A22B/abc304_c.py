import sys
from collections import deque
from itertools import combinations

def main():
    n, d = map(int, sys.stdin.readline().split())
    points_x = []
    points_y = []
    for _ in range(n):
        x, y = map(int, sys.stdin.readline().split())
        points_x.append(x)
        points_y.append(y)
    
    d_sq = d * d
    adj = [[] for _ in range(n)]
    
    for i, j in combinations(range(n), 2):
        xi = points_x[i]
        yi = points_y[i]
        xj = points_x[j]
        yj = points_y[j]
        dx = xi - xj
        dy = yi - yj
        dist_sq = dx * dx + dy * dy
        if dist_sq <= d_sq:
            adj[i].append(j)
            adj[j].append(i)
    
    visited = [False] * n
    queue = deque()
    queue.append(0)
    visited[0] = True
    
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                queue.append(v)
    
    for i in range(n):
        print("Yes" if visited[i] else "No")

if __name__ == "__main__":
    main()