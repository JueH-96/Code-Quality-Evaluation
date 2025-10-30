from collections import defaultdict

def dfs(graph, node, parent, depth, depths):
    depths[node] = depth
    for child, weight in graph[node]:
        if child != parent:
            dfs(graph, child, node, depth + weight, depths)

def solve(N, edges):
    graph = defaultdict(list)
    for u, v, l in edges:
        graph[u].append((v, l))
        graph[v].append((u, l))

    # Calculate depths of all nodes from node 1
    depths = [0] * (N + 1)
    dfs(graph, 1, -1, 0, depths)

    # Sort vertices by depth (distance from node 1)
    sorted_vertices = sorted(range(1, N + 1), key=lambda x: depths[x], reverse=True)

    # Calculate the sum of all edges (since it's a tree, it will be used often)
    total_edge_length = sum(l for _, _, l in edges)

    # For each K, find the optimal score
    answers = []
    for K in range(1, N + 1):
        # For K vertices, the optimal score is twice the sum of all edges minus
        # the sum of the K-1 deepest edges (since Takahashi will not return on these edges)
        score = 2 * total_edge_length - sum(depths[sorted_vertices[i]] for i in range(K - 1))
        answers.append(score)

    return answers

# Read input
N = int(input())
edges = [tuple(map(int, input().split())) for _ in range(N - 1)]

# Solve the problem
answers = solve(N, edges)

# Output the answers
for answer in answers:
    print(answer)