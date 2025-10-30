from collections import defaultdict

N, K = map(int, input().split())

adj = defaultdict(list)
for _ in range(N * K - 1):
    u, v = map(int, input().split())
    adj[u].append(v)
    adj[v].append(u)

if K == 1:
    print("Yes")
else:
    used = [False] * (N * K + 1)
    
    def dfs(vertex, path_length, path):
        if path_length == K:
            return True
        
        for neighbor in adj[vertex]:
            if not used[neighbor]:
                used[neighbor] = True
                path.append(neighbor)
                if dfs(neighbor, path_length + 1, path):
                    return True
                path.pop()
                used[neighbor] = False
        
        return False
    
    def solve_recursive(paths_formed):
        if paths_formed == N:
            return True
        
        # Find the first unused vertex
        start = -1
        for i in range(1, N * K + 1):
            if not used[i]:
                start = i
                break
        
        if start == -1:
            return paths_formed == N
        
        # Try to form a path starting from this vertex
        used[start] = True
        path = [start]
        if dfs(start, 1, path):
            if solve_recursive(paths_formed + 1):
                return True
        
        # Backtrack: unmark all vertices in the path
        for vertex in path:
            used[vertex] = False
        
        return False
    
    if solve_recursive(0):
        print("Yes")
    else:
        print("No")