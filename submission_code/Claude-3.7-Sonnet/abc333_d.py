from collections import defaultdict

def min_operations_to_delete_vertex_1(n, edges):
    # Build the adjacency list representation of the tree
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # Root the tree at vertex 1
    def root_tree(node, parent):
        children = []
        for neighbor in graph[node]:
            if neighbor != parent:
                children.append(neighbor)
                root_tree(neighbor, node)
        graph[node] = children
    
    root_tree(1, -1)
    
    # If vertex 1 has no children, it's already a leaf
    if not graph[1]:
        return 1
    
    # Calculate the size of each subtree
    subtree_size = {}
    
    def get_subtree_size(node):
        if not graph[node]:
            subtree_size[node] = 1
            return 1
        
        size = 1  # Count the node itself
        for child in graph[node]:
            size += get_subtree_size(child)
        subtree_size[node] = size
        return size
    
    # Calculate the size of each subtree rooted at a child of vertex 1
    for child in graph[1]:
        get_subtree_size(child)
    
    # Find the largest subtree
    largest_child = max(graph[1], key=lambda child: subtree_size.get(child, 0)) if graph[1] else None
    largest_size = subtree_size.get(largest_child, 0) if largest_child else 0
    
    # Operations = (delete all other subtrees) + (delete vertex 1)
    operations_for_other_subtrees = n - 1 - largest_size
    
    return operations_for_other_subtrees + 1

# Read input and solve the problem
n = int(input())
edges = []
for _ in range(n-1):
    u, v = map(int, input().split())
    edges.append((u, v))

print(min_operations_to_delete_vertex_1(n, edges))