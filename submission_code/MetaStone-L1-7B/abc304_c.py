import sys
from collections import deque

def main():
    N, D = map(int, sys.stdin.readline().split())
    points = []
    for _ in range(N):
        x, y = map(int, sys.stdin.readline().split())
        points.append((x, y))
    D_sq = D * D

    adj = [[] for _ in range(N)]
    for i in range(N):
        xi, yi = points[i]
        for j in range(N):
            if i == j:
                continue
            xj, yj = points[j]
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
            if dist_sq <= D_sq:
                adj[i].append(j)

    visited = [False] * N
    q = deque()
    q.append(0)
    visited[0] = True

    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                q.append(v)

    for i in range(N):
        print("Yes" if visited[i] else "No")

if __name__ == '__main__':
    main()