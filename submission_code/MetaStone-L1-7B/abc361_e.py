import sys
from collections import deque

def main():
    N = int(sys.stdin.readline())
    edges = [[] for _ in range(N + 1)]
    total_sum = 0

    for _ in range(N - 1):
        a, b, c = map(int, sys.stdin.readline().split())
        edges[a].append((b, c))
        edges[b].append((a, c))
        total_sum += c

    # Function to perform BFS and find the farthest node and its distance
    def bfs(start):
        visited = [False] * (N + 1)
        q = deque()
        q.append((start, 0))
        visited[start] = True
        max_dist = 0
        far_node = start

        while q:
            node, dist = q.popleft()
            if dist > max_dist:
                max_dist = dist
                far_node = node
            for neighbor, weight in edges[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    q.append((neighbor, dist + weight))
        return far_node, max_dist

    # First BFS to find one end of the diameter
    u, _ = bfs(1)
    # Second BFS to find the other end and the diameter
    v, diameter = bfs(u)

    answer = 2 * total_sum - diameter
    print(answer)

if __name__ == '__main__':
    main()