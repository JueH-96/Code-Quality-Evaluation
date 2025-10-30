from collections import defaultdict

def dfs(node, parent, graph, leaves):
    if len(graph[node]) == 1 and node != parent:
        leaves.add(node)
        return 1
    count = 0
    for neighbor in graph[node]:
        if neighbor != parent:
            count += dfs(neighbor, node, graph, leaves)
    return count

def min_operations_to_delete_vertex_1(N, edges):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    leaves = set()
    dfs(1, -1, graph, leaves)
    
    operations = 0
    while 1 in leaves:
        new_leaves = set()
        for leaf in leaves:
            if leaf != 1:
                for neighbor in graph[leaf]:
                    new_leaves.add(neighbor)
        leaves = new_leaves
        operations += 1
    
    return operations