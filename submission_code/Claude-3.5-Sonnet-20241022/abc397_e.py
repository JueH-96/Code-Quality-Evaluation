from collections import defaultdict

def find_paths(graph, N, K):
    NK = N * K
    visited = [False] * (NK + 1)
    paths = []
    current_path = []
    
    def dfs(vertex):
        if len(current_path) == K:
            paths.append(current_path[:])
            return len(paths) <= N
        
        visited[vertex] = True
        current_path.append(vertex)
        
        for neighbor in graph[vertex]:
            if not visited[neighbor]:
                if dfs(neighbor):
                    return True
                
        if len(current_path) == K:
            paths.append(current_path[:])
            return len(paths) <= N
            
        visited[vertex] = False
        current_path.pop()
        return False

    # Try starting from each vertex
    for start in range(1, NK + 1):
        if not visited[start]:
            current_path = []
            if dfs(start):
                if len(paths) == N and all(len(p) == K for p in paths):
                    # Check if we used all vertices
                    used = set()
                    for path in paths:
                        used.update(path)
                    if len(used) == NK:
                        return True
            # Reset for next attempt
            visited = [False] * (NK + 1)
            paths = []
    
    return False

def main():
    # Read input
    N, K = map(int, input().split())
    
    # Create adjacency list representation of the tree
    graph = defaultdict(list)
    for _ in range(N * K - 1):
        u, v = map(int, input().split())
        graph[u].append(v)
        graph[v].append(u)
    
    # Find if decomposition is possible
    result = find_paths(graph, N, K)
    
    # Print result
    print("Yes" if result else "No")

if __name__ == "__main__":
    main()