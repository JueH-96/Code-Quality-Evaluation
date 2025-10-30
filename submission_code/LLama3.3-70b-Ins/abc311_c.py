def find_cycle(n, edges):
    """
    Finds a directed cycle in a graph with n vertices and n edges.
    
    Args:
    n (int): The number of vertices in the graph.
    edges (list): A list of edges where the i-th edge goes from vertex i to vertex edges[i].
    
    Returns:
    list: A list of vertices representing a directed cycle.
    """
    # Create an adjacency list representation of the graph
    graph = {i: edges[i-1] for i in range(1, n+1)}
    
    # Initialize a set to keep track of visited vertices
    visited = set()
    
    # Initialize a list to store the cycle
    cycle = []
    
    # Define a helper function to perform DFS
    def dfs(vertex, path):
        nonlocal cycle
        # Add the current vertex to the path
        path.append(vertex)
        
        # Mark the current vertex as visited
        visited.add(vertex)
        
        # Get the next vertex in the path
        next_vertex = graph[vertex]
        
        # If the next vertex is already in the path, we've found a cycle
        if next_vertex in path:
            # Find the index of the next vertex in the path
            index = path.index(next_vertex)
            # Extract the cycle from the path
            cycle = path[index:]
            return
        
        # If the next vertex has not been visited, continue DFS
        if next_vertex not in visited:
            dfs(next_vertex, path)
        
        # Remove the current vertex from the path
        path.pop()
    
    # Perform DFS from each unvisited vertex
    for i in range(1, n+1):
        if i not in visited:
            dfs(i, [])
    
    return cycle

# Read the input
n = int(input())
edges = list(map(int, input().split()))

# Find a directed cycle
cycle = find_cycle(n, edges)

# Print the result
print(len(cycle))
print(*cycle)