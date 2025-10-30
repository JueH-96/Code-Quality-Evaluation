from collections import deque

def bfs(graph, start, end):
    """
    Performs a breadth-first search to find the shortest path between two nodes in a graph.
    
    Args:
    graph (dict): The graph represented as an adjacency list.
    start (int): The starting node.
    end (int): The ending node.
    
    Returns:
    list: The shortest path from the start node to the end node.
    """
    queue = deque([[start]])
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == end:
            return path
        for neighbor in graph[node]:
            if neighbor not in path:
                queue.append(path + [neighbor])
    return None

def find_min_tree(graph, nodes):
    """
    Finds the minimum number of vertices in a tree that includes all specified nodes.
    
    Args:
    graph (dict): The graph represented as an adjacency list.
    nodes (list): The list of nodes that must be included in the tree.
    
    Returns:
    int: The minimum number of vertices in the tree.
    """
    # Find the shortest paths between all pairs of nodes
    paths = {}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            path = bfs(graph, nodes[i], nodes[j])
            paths[(nodes[i], nodes[j])] = path
            paths[(nodes[j], nodes[i])] = path[::-1]
    
    # Find the minimum number of vertices in the tree
    min_vertices = float('inf')
    for path in paths.values():
        min_vertices = min(min_vertices, len(set(path)))
    
    return min_vertices

def main():
    # Read the input
    n, k = map(int, input().split())
    graph = {i: [] for i in range(1, n + 1)}
    for _ in range(n - 1):
        a, b = map(int, input().split())
        graph[a].append(b)
        graph[b].append(a)
    nodes = list(map(int, input().split()))
    
    # Find the minimum number of vertices in the tree
    min_vertices = find_min_tree(graph, nodes)
    
    # Print the answer
    print(min_vertices)

if __name__ == "__main__":
    main()