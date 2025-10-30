import sys
from collections import deque

def build_graph(n, edges):
    g = [[] for _ in range(n)]
    total = 0
    for a, b, c in edges:
        a -= 1  # 0-indexing
        b -= 1
        g[a].append((b, c))
        g[b].append((a, c))
        total += c
    return g, total


def farthest_from(start, graph):
    """
    Returns (distances list, index_of_farthest_vertex)
    Distances are computed by a simple stack DFS because the input is a tree,
    so every first visit already gives the unique shortest path.
    """
    n = len(graph)
    dist = [-1] * n
    stack = [start]
    dist[start] = 0
    while stack:
        v = stack.pop()
        for nxt, w in graph[v]:
            if dist[nxt] == -1:
                dist[nxt] = dist[v] + w
                stack.append(nxt)
    farthest_v = max(range(n), key=lambda i: dist[i])
    return dist, farthest_v


def main() -> None:
    input_data = sys.stdin.buffer.read().split()
    it = iter(input_data)
    n = int(next(it))
    edges = [(int(next(it)), int(next(it)), int(next(it))) for _ in range(n - 1)]

    graph, total_weight = build_graph(n, edges)

    # 1st traversal: from any node (say 0) to find one end of the diameter
    _, u = farthest_from(0, graph)

    # 2nd traversal: real diameter length
    dist_u, _ = farthest_from(u, graph)
    diameter = max(dist_u)

    # Minimum walk length that visits every city at least once
    answer = 2 * total_weight - diameter
    print(answer)


if __name__ == "__main__":
    main()