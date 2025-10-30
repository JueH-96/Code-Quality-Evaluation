import sys
from collections import deque

def main():
    N = int(sys.stdin.readline())
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        A, B, C = map(int, sys.stdin.readline().split())
        edges[A].append((B, C))
        edges[B].append((A, C))
    
    # Find the farthest node from any node (e.g., node 1)
    def bfs(start):
        dist = [0] * (N+1)
        visited = [False] * (N+1)
        q = deque()
        q.append(start)
        visited[start] = True
        while q:
            u = q.popleft()
            for v, c in edges[u]:
                if not visited[v]:
                    visited[v] = True
                    dist[v] = dist[u] + c
                    q.append(v)
        max_dist = max(dist)
        farthest = dist.index(max_dist)
        return farthest, max_dist
    
    # First BFS to find one end of the diameter
    start = 1
    farthest1, _ = bfs(start)
    # Second BFS to find the other end of the diameter and the diameter length
    farthest2, diameter = bfs(farthest1)
    
    # The minimum travel distance is 2 * total_length - diameter
    total_length = 0
    for i in range(1, N+1):
        for _, c in edges[i]:
            total_length += c
    total_length = total_length // 2  # Since each edge is counted twice
    
    min_distance = 2 * total_length - diameter
    print(min_distance)

if __name__ == "__main__":
    main()