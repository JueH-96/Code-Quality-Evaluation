from itertools import permutations

def is_strongly_connected(graph, n):
    def dfs(start, g):
        visited = [False] * (n + 1)
        stack = [start]
        visited[start] = True
        
        while stack:
            v = stack.pop()
            for neighbor in g[v]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        
        return all(visited[1:n+1])
    
    # Check if all vertices can be reached from vertex 1
    if not dfs(1, graph):
        return False
    
    # Create a reversed graph
    reversed_graph = [[] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in graph[i]:
            reversed_graph[j].append(i)
    
    # Check if all vertices can reach vertex 1
    return dfs(1, reversed_graph)

def solve(n, s):
    # Identify white and black vertices
    white = [i for i, color in enumerate(s, 1) if color == 'W']
    black = [i for i, color in enumerate(s, 1) if color == 'B']
    
    # Original graph
    graph = [[] for _ in range(2*n + 1)]
    for i in range(1, 2*n):
        graph[i].append(i + 1)
    
    count = 0
    MOD = 998244353
    
    # Try all permutations of white vertices
    for p in permutations(white):
        # Create a copy of the original graph
        temp_graph = [list(neighbors) for neighbors in graph]
        
        # Add white-to-black edges
        for i in range(n):
            temp_graph[p[i]].append(black[i])
        
        # Check for strong connectivity
        if is_strongly_connected(temp_graph, 2*n):
            count = (count + 1) % MOD
    
    return count

if __name__ == "__main__":
    n = int(input().strip())
    s = input().strip()
    
    print(solve(n, s))