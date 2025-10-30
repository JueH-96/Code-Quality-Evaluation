import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        A, B, C = map(int, sys.stdin.readline().split())
        edges[A].append((B, C))
        edges[B].append((A, C))
    
    # Find the diameter of the tree
    def bfs(start):
        dist = [-1] * (N+1)
        q = deque()
        q.append(start)
        dist[start] = 0
        while q:
            u = q.popleft()
            for v, c in edges[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + c
                    q.append(v)
        max_dist = max(dist[1:])
        farthest = dist.index(max_dist)
        return farthest, max_dist
    
    # First BFS to find one end of the diameter
    u, _ = bfs(1)
    # Second BFS to find the other end and the diameter
    v, diameter = bfs(u)
    
    # Total distance is 2 * sum of all edges - diameter
    total = 0
    for i in range(1, N+1):
        for _, c in edges[i]:
            total += c
    total = total // 2  # Since each edge is counted twice
    min_distance = 2 * total - diameter
    print(min_distance)

if __name__ == "__main__":
    main()