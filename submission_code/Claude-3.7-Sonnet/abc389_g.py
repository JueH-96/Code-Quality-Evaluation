from collections import deque
from itertools import combinations

def solve(n, p):
    results = []
    
    # Enumerate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for m in range(n-1, (n*(n-1))//2 + 1):
        count = 0
        
        # Generate all possible combinations of m edges from all_edges
        for edge_combo in combinations(all_edges, m):
            # Construct the graph as an adjacency list
            graph = [[] for _ in range(n)]
            for u, v in edge_combo:
                graph[u].append(v)
                graph[v].append(u)
            
            # Compute distances from vertex 0 (corresponding to vertex 1 in the problem)
            distances = [-1] * n
            distances[0] = 0
            queue = deque([0])
            while queue:
                vertex = queue.popleft()
                for neighbor in graph[vertex]:
                    if distances[neighbor] == -1:
                        distances[neighbor] = distances[vertex] + 1
                        queue.append(neighbor)
            
            # Check if the graph is connected and if the condition is met
            if all(distance != -1 for distance in distances):
                even_count = sum(1 for d in distances if d % 2 == 0)
                if even_count * 2 == n:  # This is equivalent to even_count == n - even_count
                    count += 1
        
        results.append(count % p)
    
    return " ".join(map(str, results))

# Read input
n, p = map(int, input().split())
print(solve(n, p))