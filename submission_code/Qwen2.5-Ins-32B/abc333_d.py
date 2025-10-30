import sys
from collections import deque, defaultdict

def solve():
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    edges = [(int(data[i]), int(data[i+1])) for i in range(1, 2*n-1, 2)]
    
    graph = defaultdict(set)
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)
    
    queue = deque([node for node in graph if node != 1 and len(graph[node]) == 1])
    operations = 0
    
    while queue:
        for _ in range(len(queue)):
            node = queue.popleft()
            if node == 1:
                return operations
            for neighbor in graph[node]:
                graph[neighbor].remove(node)
                if len(graph[neighbor]) == 1:
                    queue.append(neighbor)
        operations += 1
    
    return operations

print(solve())