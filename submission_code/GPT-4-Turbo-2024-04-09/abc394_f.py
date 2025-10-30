import sys
from collections import defaultdict, deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
edges = [(int(data[i*2+1]), int(data[i*2+2])) for i in range(N-1)]

# Adjacency list representation of the graph
graph = defaultdict(list)
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)

# To find the largest alkane subgraph, we need to find a subtree where all nodes have degree 1 or 4
# and at least one node has degree 4.

def bfs_find_farthest(node):
    """ Perform BFS and return the farthest node and its distance from the start node """
    queue = deque([node])
    distances = {node: 0}
    farthest_node = node
    max_distance = 0
    
    while queue:
        current = queue.popleft()
        current_distance = distances[current]
        
        for neighbor in graph[current]:
            if neighbor not in distances:
                distances[neighbor] = current_distance + 1
                queue.append(neighbor)
                if distances[neighbor] > max_distance:
                    max_distance = distances[neighbor]
                    farthest_node = neighbor
    
    return farthest_node, max_distance

def find_diameter():
    """ Find the diameter of the tree using two BFS runs """
    # Start from any node, here we start from node 1 (or any arbitrary node)
    farthest_from_start, _ = bfs_find_farthest(1)
    # Use the farthest node found and perform another BFS to find the diameter
    farthest_from_farthest, diameter = bfs_find_farthest(farthest_from_start)
    return diameter

def count_valid_alkane_subgraph():
    # We need to find a subtree where all nodes have degree 1 or 4 and at least one node has degree 4.
    # We will use a BFS to determine the largest such subtree.
    
    # First, we need to find all nodes with degree 1 or 4
    degree = defaultdict(int)
    for u in graph:
        degree[u] = len(graph[u])
    
    # We need to find the largest connected component of the graph where all nodes have degree 1 or 4
    # and at least one node has degree 4.
    visited = set()
    max_size = 0
    found_degree_4 = False
    
    def bfs_component(start):
        if start in visited:
            return 0, False
        queue = deque([start])
        visited.add(start)
        component_size = 0
        has_degree_4 = False
        
        while queue:
            node = queue.popleft()
            component_size += 1
            if degree[node] == 4:
                has_degree_4 = True
            
            for neighbor in graph[node]:
                if neighbor not in visited and (degree[neighbor] == 1 or degree[neighbor] == 4):
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return component_size, has_degree_4
    
    for node in range(1, N+1):
        if degree[node] == 1 or degree[node] == 4:
            size, has_4 = bfs_component(node)
            if has_4:
                max_size = max(max_size, size)
                found_degree_4 = True
    
    if found_degree_4:
        return max_size
    else:
        return -1

# Output the result
print(count_valid_alkane_subgraph())