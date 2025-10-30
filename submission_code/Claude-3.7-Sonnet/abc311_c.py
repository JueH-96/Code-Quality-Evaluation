def main():
    n = int(input().strip())
    A = list(map(int, input().strip().split()))
    
    # Construct the graph
    graph = {i+1: A[i] for i in range(n)}
    
    # Find a cycle
    cycle = find_cycle(graph, n)
    
    # Print the output
    print(len(cycle))
    print(" ".join(map(str, cycle)))

def find_cycle(graph, n):
    """
    Find a cycle in a directed graph.
    Returns the cycle as a list of vertices.
    """
    visited = {}  # Maps node to its position in the traversal
    
    def dfs(start_node):
        pos = 0
        path = []
        node = start_node
        
        while node not in visited:
            visited[node] = pos
            path.append(node)
            node = graph[node]
            pos += 1
        
        # Found a cycle
        cycle_start = visited[node]
        return path[cycle_start:]
    
    for node in range(1, n + 1):
        if node not in visited:
            cycle = dfs(node)
            if cycle:
                return cycle
    
    return None  # No cycle found (but the problem guarantees one exists)

if __name__ == "__main__":
    main()