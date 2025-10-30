import sys
from collections import deque

def main():
    N, M, S, T = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N+1)]
    edges = []  # Store edges in both directions
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
        edges.append((u, v))
        edges.append((v, u))  # Both directions for edge tracking

    # BFS to find shortest path from S to T and track edges used
    dist_S = [-1] * (N + 1)
    prev_S = [-1] * (N + 1)
    q = deque()
    q.append(S)
    dist_S[S] = 0
    found = False
    while q:
        u = q.popleft()
        if u == T:
            found = True
            break
        for v in adj[u]:
            if dist_S[v] == -1:
                dist_S[v] = dist_S[u] + 1
                prev_S[v] = u
                q.append(v)
    if not found:
        print(-1)
        return

    # Reconstruct the path edges
    path_edges = set()
    current = T
    while current != S:
        prev_node = prev_S[current]
        path_edges.add((prev_node, current))
        path_edges.add((current, prev_node))  # Both directions
        current = prev_node

    # BFS from T to S excluding the edges in path_edges
    dist_T = [-1] * (N + 1)
    q = deque()
    q.append(T)
    dist_T[T] = 0
    while q:
        u = q.popleft()
        if u == S:
            break
        for v in adj[u]:
            edge = (u, v)
            if edge not in path_edges and dist_T[v] == -1:
                dist_T[v] = dist_T[u] + 1
                q.append(v)
    if dist_T[S] == -1:
        print(-1)
    else:
        print(dist_S[T] + dist_T[S])

if __name__ == "__main__":
    main()