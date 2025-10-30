import sys
from collections import defaultdict, deque

def solve():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    edges = [tuple(map(int, data[i].split())) for i in range(1, N)]
    
    # Build the graph
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # Initialize the queue with all leaf nodes
    queue = deque([node for node in graph if len(graph[node]) == 1])
    visited = set()
    
    # Perform the operations
    operations = 0
    while queue:
        for _ in range(len(queue)):
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            if node == 1:
                print(operations)
                return
            for neighbor in graph[node]:
                if neighbor not in visited:
                    graph[neighbor].remove(node)
                    if len(graph[neighbor]) == 1:
                        queue.append(neighbor)
        operations += 1

solve()