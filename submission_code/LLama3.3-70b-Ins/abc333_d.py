from collections import defaultdict

def dfs(graph, node, parent):
    """Perform DFS traversal from the given node."""
    # Initialize the count of nodes to delete
    count = 0
    
    # Iterate over all neighbors of the current node
    for neighbor in graph[node]:
        # Skip the parent node to avoid revisiting it
        if neighbor == parent:
            continue
        
        # Recursively perform DFS traversal from the neighbor
        count += dfs(graph, neighbor, node)
    
    # If the current node is a leaf node (i.e., it has no children or only one child),
    # increment the count by 1
    if len(graph[node]) == 1 or (len(graph[node]) == 2 and parent in graph[node]):
        count += 1
    
    return count

def solve():
    """Solve the problem."""
    # Read the number of nodes
    n = int(input())
    
    # Create an adjacency list representation of the graph
    graph = defaultdict(list)
    for _ in range(n - 1):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)
    
    # Perform DFS traversal from node 1 and print the result
    print(dfs(graph, 1, -1))

if __name__ == "__main__":
    solve()