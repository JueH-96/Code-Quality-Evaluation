# YOUR CODE HERE
def find_min_ng_list_count(N, product_names):
    from collections import defaultdict

    # Create a graph where each node is a letter and edges are product names
    graph = defaultdict(list)
    for name in product_names:
        graph[name[0]].append(name[1])

    # Function to perform DFS and find all connected components
    def dfs(node, visited, component):
        visited.add(node)
        component.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, visited, component)

    visited = set()
    components = []

    # Find all connected components in the graph
    for node in graph:
        if node not in visited:
            component = []
            dfs(node, visited, component)
            components.append(component)

    # The number of connected components is the minimum number of NG strings needed
    return len(components)

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    product_names = data[1:]
    result = find_min_ng_list_count(N, product_names)
    print(result)