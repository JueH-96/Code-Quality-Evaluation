# YOUR CODE HERE
from collections import defaultdict

def dfs(graph, node, parent):
    children = [child for child in graph[node] if child != parent]
    if not children:
        return 1, 0  # leaf node
    
    child_depths = [dfs(graph, child, node) for child in children]
    
    if all(d == child_depths[0] for d, _ in child_depths):
        return child_depths[0][0] + 1, len(children)
    
    max_depth, _ = max(child_depths)
    return max_depth, 0

def solve(N, edges):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    best_deleted = N
    for root in range(1, N+1):
        depth, x = dfs(graph, root, -1)
        if x > 0:
            y = depth - 2
            snowflake_size = 1 + x + x * y
            deleted = N - snowflake_size
            best_deleted = min(best_deleted, deleted)
    
    return best_deleted

# Read input
N = int(input())
edges = [list(map(int, input().split())) for _ in range(N-1)]

# Solve and print output
print(solve(N, edges))