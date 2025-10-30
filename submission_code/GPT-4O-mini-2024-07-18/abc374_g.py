def min_ng_list_size(N, product_names):
    # Create a set of used product names for quick lookup
    used_names = set(product_names)
    
    # Create a graph where each node is a product name
    from collections import defaultdict
    
    graph = defaultdict(set)
    
    # Build the graph based on the first and second letters of the product names
    for name in used_names:
        first, second = name[0], name[1]
        graph[first].add(second)
    
    # To find the minimum number of strings in the NG list, we need to find strongly connected components
    visited = set()
    components = []
    
    def dfs(node, component):
        visited.add(node)
        component.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, component)
    
    # Find all components
    for letter in graph:
        if letter not in visited:
            component = []
            dfs(letter, component)
            components.append(component)
    
    # Each component can be represented by a single string in the NG list
    return len(components)

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
product_names = data[1:N+1]

# Calculate the result
result = min_ng_list_size(N, product_names)

# Print the result
print(result)