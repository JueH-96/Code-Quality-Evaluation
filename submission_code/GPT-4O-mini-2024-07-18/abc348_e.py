def min_f_of_tree(N, edges, C):
    from collections import defaultdict, deque
    
    # Build the tree as an adjacency list
    tree = defaultdict(list)
    for a, b in edges:
        tree[a].append(b)
        tree[b].append(a)
    
    # Calculate initial f(1)
    def dfs(node, parent):
        total_cost = 0
        total_weight = 0
        for neighbor in tree[node]:
            if neighbor != parent:
                child_cost, child_weight = dfs(neighbor, node)
                total_cost += child_cost + child_weight
                total_weight += child_weight + C[neighbor - 1]
        return total_cost, total_weight

    # First calculate f(1)
    initial_cost, total_weight = dfs(1, -1)
    f_values = [0] * (N + 1)
    f_values[1] = initial_cost

    # Calculate f(v) for all v using the relationship
    for v in range(2, N + 1):
        f_values[v] = f_values[v - 1] + total_weight - 2 * C[v - 1]
    
    # Find the minimum f(v)
    return min(f_values[1:])

import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
edges = [tuple(map(int, line.split())) for line in data[1:N]]
C = list(map(int, data[N].split()))

result = min_f_of_tree(N, edges, C)
print(result)