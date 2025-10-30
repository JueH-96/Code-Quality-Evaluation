import sys
from collections import deque

def main():
    n = int(sys.stdin.readline())
    adj = [[] for _ in range(n + 1)]
    sum_c = 0
    for _ in range(n - 1):
        a, b, c = map(int, sys.stdin.readline().split())
        adj[a].append((b, c))
        adj[b].append((a, c))
        sum_c += c

    def bfs(start):
        visited = [False] * (n + 1)
        q = deque()
        q.append((start, 0))
        visited[start] = True
        max_dist = 0
        far_node = start
        while q:
            current, dist = q.popleft()
            for neighbor, cost in adj[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    new_dist = dist + cost
                    if new_dist > max_dist:
                        max_dist = new_dist
                        far_node = neighbor
                    q.append((neighbor, new_dist))
        return far_node, max_dist

    # First BFS to find one end of the diameter
    u, _ = bfs(1)
    # Second BFS to find the other end and the diameter
    v, diameter = bfs(u)
    print(2 * sum_c - diameter)

if __name__ == "__main__":
    main()