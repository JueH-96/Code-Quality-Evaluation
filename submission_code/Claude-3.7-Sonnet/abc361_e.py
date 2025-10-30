from collections import deque

def main():
    n = int(input())
    
    # Create adjacency list for the graph
    graph = [[] for _ in range(n + 1)]
    total_length = 0
    
    # Process input and build the graph
    for _ in range(n - 1):
        a, b, c = map(int, input().split())
        graph[a].append((b, c))
        graph[b].append((a, c))
        total_length += c
    
    # Find the diameter of the tree
    # First, find the farthest node from an arbitrary node (node 1)
    farthest_node, _ = find_farthest_node(graph, 1, n)
    
    # Then, find the farthest node from the farthest node to get the diameter
    _, diameter = find_farthest_node(graph, farthest_node, n)
    
    # The minimum travel distance is twice the sum of all road lengths minus the diameter
    answer = 2 * total_length - diameter
    
    print(answer)

def find_farthest_node(graph, start, n):
    """
    Find the farthest node from the starting node and its distance
    using BFS.
    """
    distances = [-1] * (n + 1)
    distances[start] = 0
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        
        for neighbor, length in graph[node]:
            if distances[neighbor] == -1:
                distances[neighbor] = distances[node] + length
                queue.append(neighbor)
    
    # Find the node with maximum distance
    farthest_node = 1
    for i in range(1, n + 1):
        if distances[i] > distances[farthest_node]:
            farthest_node = i
    
    return farthest_node, distances[farthest_node]

if __name__ == "__main__":
    main()