def find_min_xor_path(n, m, edges):
    from collections import defaultdict
    
    # Create adjacency list
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))
    
    # To store the minimum XOR found
    min_xor = float('inf')
    
    # DFS function
    def dfs(node, current_xor, visited):
        nonlocal min_xor
        if node == n:
            min_xor = min(min_xor, current_xor)
            return
        
        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                dfs(neighbor, current_xor ^ weight, visited)
                visited.remove(neighbor)
    
    # Start DFS from node 1
    visited = set()
    visited.add(1)
    dfs(1, 0, visited)
    
    return min_xor

# Reading input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
edges = []

index = 2
for _ in range(M):
    u = int(data[index])
    v = int(data[index + 1])
    w = int(data[index + 2])
    edges.append((u, v, w))
    index += 3

# Find the minimum XOR path
result = find_min_xor_path(N, M, edges)
print(result)