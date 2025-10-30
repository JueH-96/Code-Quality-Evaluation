def min_operations_to_delete_vertex_one(N, edges):
    from collections import defaultdict, deque

    # Build the adjacency list for the tree
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    # If vertex 1 is a leaf, we can delete it immediately
    if len(graph[1]) == 1:
        return 1

    # BFS to find the depth of each node from vertex 1
    depth = {1: 0}
    queue = deque([1])
    
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in depth:  # Not visited
                depth[neighbor] = depth[node] + 1
                queue.append(neighbor)

    # The minimum number of operations required to delete vertex 1
    # is equal to the maximum depth of any leaf node in the tree
    max_depth = max(depth.values())
    
    return max_depth

import sys
input = sys.stdin.read

data = input().splitlines()
N = int(data[0])
edges = [tuple(map(int, line.split())) for line in data[1:N]]

result = min_operations_to_delete_vertex_one(N, edges)
print(result)