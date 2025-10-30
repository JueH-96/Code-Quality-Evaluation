from collections import deque
import sys

def main():
    n, d = map(int, sys.stdin.readline().split())
    people = [tuple(map(int, sys.stdin.readline().split())) for _ in range(n)]
    D_squared = d * d

    adj = [[] for _ in range(n)]
    for i in range(n):
        xi, yi = people[i]
        for j in range(i + 1, n):
            xj, yj = people[j]
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
            if dist_sq <= D_squared:
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    q = deque()
    q.append(0)
    visited[0] = True

    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                q.append(v)

    for status in visited:
        print("Yes" if status else "No")

if __name__ == "__main__":
    main()