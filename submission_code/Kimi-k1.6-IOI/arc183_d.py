import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    edges = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        a = int(input[idx])
        b = int(input[idx + 1])
        edges[a].append(b)
        edges[b].append(a)
        idx += 2
    
    # Compute bipartition
    color = [-1] * (N + 1)
    q = deque()
    q.append(1)
    color[1] = 0
    while q:
        u = q.popleft()
        for v in edges[u]:
            if color[v] == -1:
                color[v] = color[u] ^ 1
                q.append(v)
    
    # Function to perform BFS and return farthest node and distance array
    def bfs(start):
        dist = [-1] * (N + 1)
        q = deque()
        q.append(start)
        dist[start] = 0
        farthest = start
        max_d = 0
        while q:
            u = q.popleft()
            for v in edges[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
                    if dist[v] > max_d:
                        max_d = dist[v]
                        farthest = v
        return farthest, dist
    
    # Find diameter endpoints
    u, _ = bfs(1)
    v, _ = bfs(u)
    
    # Compute distances from u and v
    _, dist_u = bfs(u)
    _, dist_v = bfs(v)
    
    # Compute max_dist for each node
    max_dist = [0] * (N + 1)
    for i in range(1, N + 1):
        max_dist[i] = max(dist_u[i], dist_v[i])
    
    # Split into black and white lists with negative max_dist for descending sort
    black = []
    white = []
    for i in range(1, N + 1):
        if color[i] == 0:
            black.append((-max_dist[i], i))
        else:
            white.append((-max_dist[i], i))
    
    # Sort both lists
    black.sort()
    white.sort()
    
    # Pair the nodes
    pairs = []
    for i in range(len(black)):
        x = black[i][1]
        y = white[i][1]
        pairs.append((x, y))
    
    # Print the pairs
    for x, y in pairs:
        print(x, y)

if __name__ == '__main__':
    main()