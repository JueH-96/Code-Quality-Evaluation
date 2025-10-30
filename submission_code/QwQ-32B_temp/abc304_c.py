import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    D = int(input[idx])
    idx += 1
    d_sq = D * D
    positions = [() * 2 for _ in range(N + 1)]  # 1-based indexing

    for i in range(1, N + 1):
        x = int(input[idx])
        idx += 1
        y = int(input[idx])
        idx += 1
        positions[i] = (x, y)

    # Build adjacency list
    adj = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        xi, yi = positions[i]
        for j in range(i + 1, N + 1):
            xj, yj = positions[j]
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
            if dist_sq <= d_sq:
                adj[i].append(j)
                adj[j].append(i)

    # BFS to find all reachable nodes from 1
    visited = [False] * (N + 1)
    q = deque()
    visited[1] = True
    q.append(1)
    while q:
        current = q.popleft()
        for neighbor in adj[current]:
            if not visited[neighbor]:
                visited[neighbor] = True
                q.append(neighbor)

    for i in range(1, N + 1):
        print("Yes" if visited[i] else "No")

if __name__ == "__main__":
    main()