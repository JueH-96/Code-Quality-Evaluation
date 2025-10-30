import sys
from collections import deque

def bfs(start, adj, N):
    dist = [-1] * (N + 1)
    q = deque()
    q.append(start)
    dist[start] = 0
    while q:
        u = q.popleft()
        for v, c in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + c
                q.append(v)
    max_dist = -1
    farthest_node = start
    for i in range(1, N + 1):
        if dist[i] > max_dist:
            max_dist = dist[i]
            farthest_node = i
    return farthest_node, max_dist

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    adj = [[] for _ in range(N + 1)]
    sum_edges = 0
    for _ in range(N - 1):
        A = int(input[idx])
        B = int(input[idx + 1])
        C = int(input[idx + 2])
        idx += 3
        adj[A].append((B, C))
        adj[B].append((A, C))
        sum_edges += C
    u, _ = bfs(1, adj, N)
    v, diameter = bfs(u, adj, N)
    minimal_travel = 2 * sum_edges - diameter
    print(minimal_travel)

if __name__ == "__main__":
    main()