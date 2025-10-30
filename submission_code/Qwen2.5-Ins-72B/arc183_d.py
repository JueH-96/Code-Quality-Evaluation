import sys
from collections import defaultdict, deque

def solve():
    N = int(input())
    edges = [tuple(map(int, input().split())) for _ in range(N - 1)]
    
    # Build the graph
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # Initialize the leaves
    leaves = deque()
    for i in range(1, N + 1):
        if len(graph[i]) == 1:
            leaves.append(i)
    
    # Function to find the distance between two leaves
    def distance(u, v):
        visited = set()
        queue = deque([(u, 0)])
        while queue:
            node, dist = queue.popleft()
            if node == v:
                return dist
            if node not in visited:
                visited.add(node)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1))
        return -1
    
    # Perform the operations
    operations = []
    for _ in range(N // 2):
        u, v = leaves.popleft(), leaves.popleft()
        operations.append((u, v))
        
        # Remove the leaves from the graph
        graph[u].remove(v)
        graph[v].remove(u)
        
        # Update the leaves
        for node in [u, v]:
            for neighbor in graph[node]:
                if len(graph[neighbor]) == 1:
                    leaves.append(neighbor)
    
    # Print the operations
    for u, v in operations:
        print(u, v)

solve()