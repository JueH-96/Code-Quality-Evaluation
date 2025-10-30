from collections import defaultdict

def solve():
    # Read input
    N = int(input())
    
    # Build adjacency list representation
    adj = defaultdict(list)
    for _ in range(N-1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)
    
    # Function to find minimum operations to delete vertex 1
    def min_ops():
        # If vertex 1 is a leaf, return 1
        if len(adj[1]) == 1:
            return 1
            
        # Do BFS from vertex 1 to find levels
        levels = [-1] * (N+1)
        levels[1] = 0
        queue = [1]
        max_level = 0
        
        while queue:
            v = queue.pop(0)
            for u in adj[v]:
                if levels[u] == -1:
                    levels[u] = levels[v] + 1
                    max_level = max(max_level, levels[u])
                    queue.append(u)
        
        # Count vertices at each level
        level_count = [0] * (max_level + 1)
        for i in range(1, N+1):
            if levels[i] != -1:
                level_count[levels[i]] += 1
        
        # Process from leaves up
        ops = 0
        for level in range(max_level, -1, -1):
            if level == 0:  # vertex 1
                if level_count[level] == 1:
                    ops += 1
            else:
                ops += level_count[level]
                
        return ops
    
    print(min_ops())

solve()