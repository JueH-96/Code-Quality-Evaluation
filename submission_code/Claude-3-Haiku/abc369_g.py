from collections import defaultdict
from itertools import combinations

def solve(N, edges):
    graph = defaultdict(list)
    for u, v, l in edges:
        graph[u].append((v, l))
        graph[v].append((u, l))

    def dfs(node, parent, visited):
        visited.add(node)
        total_length = 0
        for neighbor, length in graph[node]:
            if neighbor != parent:
                total_length += length + dfs(neighbor, node, visited)
        return total_length

    def min_score(k):
        min_score = float('inf')
        for vertices in combinations(range(1, N+1), k):
            visited = set()
            total_length = dfs(1, None, visited)
            for v in vertices:
                if v not in visited:
                    total_length += 2 * dfs(v, None, visited)
            min_score = min(min_score, total_length)
        return min_score

    return [min_score(i) for i in range(1, N+1)]

# Read input
N = int(input())
edges = []
for _ in range(N-1):
    u, v, l = map(int, input().split())
    edges.append((u, v, l))

# Solve the problem
result = solve(N, edges)
for score in result:
    print(score)