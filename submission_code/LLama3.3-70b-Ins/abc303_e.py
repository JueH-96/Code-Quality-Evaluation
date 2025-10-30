from collections import defaultdict
from sys import stdin, stdout

def dfs(node, parent, graph, degrees):
    """Perform DFS traversal and calculate degrees of nodes."""
    for neighbor in graph[node]:
        if neighbor != parent:
            dfs(neighbor, node, graph, degrees)
            degrees[node] += 1

def solve():
    """Solve the problem."""
    n = int(stdin.readline().strip())
    graph = defaultdict(list)
    for _ in range(n - 1):
        u, v = map(int, stdin.readline().strip().split())
        graph[u].append(v)
        graph[v].append(u)

    degrees = [0] * (n + 1)
    dfs(1, -1, graph, degrees)

    # Find leaf nodes (nodes with degree 1)
    leaves = [i for i in range(1, n + 1) if degrees[i] == 1]

    # Find internal nodes (nodes with degree > 1)
    internals = [i for i in range(1, n + 1) if degrees[i] > 1]

    # Initialize result list
    result = []

    # Process internal nodes
    for node in internals:
        # Find the number of leaf nodes connected to the current internal node
        leaf_count = sum(1 for neighbor in graph[node] if degrees[neighbor] == 1)
        if leaf_count > 0:
            result.append(leaf_count)

    # Sort the result list
    result.sort()

    # Print the result
    print(' '.join(map(str, result)))

if __name__ == '__main__':
    solve()