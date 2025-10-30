def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    product_names = data[1:]
    
    # Create a graph where each node is a product name and there is an edge between
    # two nodes if they can be directly connected without forming an invalid product name.
    from collections import defaultdict, deque
    
    # Graph adjacency list
    graph = defaultdict(list)
    
    # Set of all product names for quick lookup
    product_set = set(product_names)
    
    # Create nodes based on the first and last letter of each product name
    for name in product_names:
        first, last = name[0], name[1]
        
        # Possible connections from this name
        # name -> XY, where Y is the first letter of another name
        for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            next_name = last + char
            if next_name in product_set:
                graph[name].append(next_name)
    
    # We need to find the minimum number of connected components in this graph
    # which can be done using a BFS or DFS to count the number of connected components.
    visited = set()
    
    def bfs(start):
        queue = deque([start])
        visited.add(start)
        while queue:
            current = queue.popleft()
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    # Count connected components
    num_components = 0
    for name in product_names:
        if name not in visited:
            bfs(name)
            num_components += 1
    
    # Output the number of connected components which is the answer
    print(num_components)