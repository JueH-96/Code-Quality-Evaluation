def main():
    import sys
    from collections import deque

    input = sys.stdin.readline

    N1, N2, M = map(int, input().split())

    # Adjacency lists for the two disconnected components
    adj1 = [[] for _ in range(N1)]
    adj2 = [[] for _ in range(N2)]

    for _ in range(M):
        a, b = map(int, input().split())
        # Edges either lie entirely in [1..N1] or entirely in [N1+1..N1+N2].
        # Because 1 and N1+N2 are disconnected, there are no cross edges.
        if b <= N1:
            # Both ends in the first subgraph
            adj1[a - 1].append(b - 1)
            adj1[b - 1].append(a - 1)
        else:
            # Both ends in the second subgraph
            adj2[a - N1 - 1].append(b - N1 - 1)
            adj2[b - N1 - 1].append(a - N1 - 1)

    def bfs(start, graph):
        dist = [-1] * len(graph)
        dist[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nxt in graph[node]:
                if dist[nxt] < 0:
                    dist[nxt] = dist[node] + 1
                    queue.append(nxt)
        return dist

    # BFS in the first subgraph from vertex 1 (local index 0)
    dist1 = bfs(0, adj1)
    max_dist1 = max(dist1)

    # BFS in the second subgraph from vertex (N1+N2) (local index N2-1)
    dist2 = bfs(N2 - 1, adj2)
    max_dist2 = max(dist2)

    # The maximum possible shortest path = max_dist1 + 1 (new edge) + max_dist2
    print(max_dist1 + max_dist2 + 1)

# Do not forget to call main()
if __name__ == "__main__":
    main()