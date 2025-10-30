import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    D = int(data[1])
    D_sq = D * D
    people = []
    idx = 2
    for _ in range(N):
        x = int(data[idx])
        y = int(data[idx + 1])
        people.append((x, y))
        idx += 2

    # Build adjacency list
    adj = [[] for _ in range(N)]
    for i in range(N):
        xi, yi = people[i]
        for j in range(i + 1, N):
            xj, yj = people[j]
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
            if dist_sq <= D_sq:
                adj[i].append(j)
                adj[j].append(i)

    # BFS to find all infected persons
    visited = [False] * N
    queue = deque()
    queue.append(0)
    visited[0] = True

    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                queue.append(v)

    # Output the results
    for i in range(N):
        print("Yes" if visited[i] else "No")

if __name__ == "__main__":
    main()