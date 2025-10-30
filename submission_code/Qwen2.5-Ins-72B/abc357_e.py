# YOUR CODE HERE
import sys
from collections import defaultdict, deque

def count_reachable_pairs(n, edges):
    # Create the graph
    graph = defaultdict(list)
    for i, edge in enumerate(edges, start=1):
        graph[i].append(edge)
    
    # Function to perform BFS and count reachable nodes
    def bfs(start):
        visited = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        return len(visited)
    
    # Count the number of reachable pairs
    total_pairs = 0
    for i in range(1, n + 1):
        total_pairs += bfs(i)
    
    return total_pairs

# Read input
n = int(input())
edges = list(map(int, input().split()))

# Compute and print the result
print(count_reachable_pairs(n, edges))