def solve():
    n = int(input())
    edges = []
    for _ in range(n - 1):
        u, v = map(int, input().split())
        edges.append((u, v))

    adj = [[] for _ in range(n + 1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    degrees = [len(adj[i]) for i in range(n + 1)]
    
    leaves = []
    for i in range(1, n + 1):
        if degrees[i] == 1:
            leaves.append(i)

    levels = []
    visited = [False] * (n + 1)

    def find_star(leaf):
        nonlocal visited
        
        q = [leaf]
        visited[leaf] = True
        star_leaves = [leaf]
        
        while q:
            u = q.pop(0)
            
            for v in adj[u]:
                if degrees[v] > 1:
                    center = v
                    break
            else:
                continue
            break
        else:
            return None
        
        q = [center]
        visited[center] = True
        
        count = 0
        
        for neighbor in adj[center]:
            if not visited[neighbor]:
                q.append(neighbor)
                visited[neighbor] = True
                count += 1
                
        return count
    
    for leaf in leaves:
        if not visited[leaf]:
            level = find_star(leaf)
            if level is not None:
                levels.append(level)
    
    levels.sort()
    print(*levels)

solve()