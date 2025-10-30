from collections import defaultdict

def dfs(node, parent, graph, distances):
    for neighbor in graph[node]:
        if neighbor != parent:
            distances[neighbor] = distances[node] + 1
            dfs(neighbor, node, graph, distances)

def solve():
    N = int(input())
    graph = defaultdict(list)
    for _ in range(N - 1):
        a, b = map(int, input().split())
        graph[a].append(b)
        graph[b].append(a)
    C = list(map(int, input().split()))

    min_f = float('inf')
    for v in range(1, N + 1):
        distances = [0] * (N + 1)
        dfs(v, -1, graph, distances)
        f = sum(C[i] * distances[i + 1] for i in range(N))
        min_f = min(min_f, f)

    print(min_f)

solve()