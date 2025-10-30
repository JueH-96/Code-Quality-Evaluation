import sys

def can_decompose_to_paths(N, K, edges):
    # Create adjacency list
    adj_list = [[] for _ in range(N * K)]
    for u, v in edges:
        adj_list[u - 1].append(v - 1)
        adj_list[v - 1].append(u - 1)

    # Perform DFS to check if the tree can be decomposed
    visited = [False] * (N * K)
    paths = []

    def dfs(node, path):
        visited[node] = True
        path.append(node)

        if len(path) == K:
            paths.append(path[:])
            path.pop()
            return

        for neighbor in adj_list[node]:
            if not visited[neighbor]:
                dfs(neighbor, path)

        path.pop()

    for i in range(N * K):
        if not visited[i]:
            path = []
            dfs(i, path)
            if len(path) != 0:
                return "No"

    if len(paths) == N:
        return "Yes"
    else:
        return "No"

# Read input from stdin
N, K = map(int, input().split())
edges = []
for _ in range(N * K - 1):
    u, v = map(int, input().split())
    edges.append((u, v))

# Solve the problem and write the output to stdout
print(can_decompose_to_paths(N, K, edges))