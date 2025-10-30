def min_operations_to_transform(N, S, T):
    from collections import defaultdict
    
    # Graph to store the mapping relations
    graph = defaultdict(set)
    
    # Build the graph of transformations needed
    for s_char, t_char in zip(S, T):
        if s_char != t_char:
            graph[s_char].add(t_char)
    
    # To find the minimum number of operations, we need to find the number of connected components
    # in the graph where each component represents a group of transformations that are interconnected.
    
    # Visited set to keep track of visited nodes
    visited = set()
    
    def dfs(node):
        stack = [node]
        component = set()
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                component.add(current)
                stack.extend(graph[current])
        return component
    
    # Count the number of connected components
    components = []
    for char in graph:
        if char not in visited:
            component = dfs(char)
            components.append(component)
    
    # Each component can be resolved with len(component) - 1 operations
    operations = 0
    for component in components:
        operations += len(component) - 1
    
    return operations

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]
T = data[2]

result = min_operations_to_transform(N, S, T)
print(result)