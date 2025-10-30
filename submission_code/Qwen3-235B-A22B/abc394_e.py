import sys
import string
from collections import deque, defaultdict

def main():
    N = int(sys.stdin.readline().strip())
    C = []
    for _ in range(N):
        line = sys.stdin.readline().strip()
        C.append(list(line))
    
    # Precompute incoming and outgoing edges for each character
    in_edges = defaultdict(lambda: defaultdict(list))  # in_edges[c][v] = list of u with u -> v
    out_edges = defaultdict(lambda: defaultdict(list)) # out_edges[c][u] = list of v with u -> v

    for u in range(N):
        for v in range(N):
            c = C[u][v]
            if c != '-':
                out_edges[c][u].append(v)
    
    for v in range(N):
        for u in range(N):
            c = C[u][v]
            if c != '-':
                in_edges[c][v].append(u)
    
    INF = float('inf')
    dist = [[INF] * N for _ in range(N)]
    
    # Initialize distance for u == v (0) and edges (1)
    for u in range(N):
        dist[u][u] = 0
    
    for u in range(N):
        for v in range(N):
            if C[u][v] != '-':
                if dist[u][v] > 1:
                    dist[u][v] = 1
    
    q = deque()
    for u in range(N):
        for v in range(N):
            if dist[u][v] == 0:
                q.append((u, v))
    for u in range(N):
        for v in range(N):
            if dist[u][v] == 1:
                q.append((u, v))
    
    # BFS
    while q:
        u, v = q.popleft()
        current_dist = dist[u][v]
        for c in string.ascii_lowercase:
            # Get all x with x -> u labeled c
            x_list = in_edges.get(c, {}).get(u, [])
            # Get all y with v -> y labeled c
            y_list = out_edges.get(c, {}).get(v, [])
            for x in x_list:
                for y in y_list:
                    if dist[x][y] > current_dist + 2:
                        dist[x][y] = current_dist + 2
                        q.append((x, y))
    
    # Output
    for u in range(N):
        row = []
        for v in range(N):
            if dist[u][v] == INF:
                row.append('-1')
            else:
                row.append(str(dist[u][v]))
        print(' '.join(row))

if __name__ == '__main__':
    main()