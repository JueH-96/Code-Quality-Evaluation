def find_reachable(N, adj):
    # Create adjacency list representation
    graph = [[] for _ in range(N+1)]
    for i in range(1, N+1):
        graph[i].append(adj[i-1])
    
    # For each vertex, find all reachable vertices using DFS
    def dfs(start):
        visited = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                for next_node in graph[node]:
                    stack.append(next_node)
        return visited
    
    # Count total reachable pairs
    total = 0
    for i in range(1, N+1):
        reachable = dfs(i)
        total += len(reachable)
    
    return total

# Read input
N = int(input())
a = list(map(int, input().split()))

# Print result
print(find_reachable(N, a))