from collections import defaultdict

def solve():
    N, M = map(int, input().split())
    edges = []
    for _ in range(M):
        u, v = map(int, input().split())
        edges.append((u, v))
    W = list(map(int, input().split()))
    A = list(map(int, input().split()))

    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    total_ops = 0
    while sum(A) > 0:
        # Find the vertex with the highest weight that has a piece
        max_weight = 0
        max_vertex = None
        for i in range(N):
            if A[i] > 0 and W[i] > max_weight:
                max_weight = W[i]
                max_vertex = i + 1

        # Remove a piece from the max vertex
        A[max_vertex - 1] -= 1
        total_ops += 1

        # Find the set of adjacent vertices with total weight less than the max vertex
        adj_vertices = graph[max_vertex]
        valid_vertices = [v for v in adj_vertices if sum(W[v-1] for v in adj_vertices if v != max_vertex) < W[max_vertex-1]]

        # Place a piece on each valid adjacent vertex
        for v in valid_vertices:
            A[v-1] += 1

    return total_ops

print(solve())