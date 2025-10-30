def solve():
    n = int(input())
    s = input()
    
    whites = []
    blacks = []
    for i in range(2 * n):
        if s[i] == 'W':
            whites.append(i + 1)
        else:
            blacks.append(i + 1)
    
    def is_strongly_connected(pairs):
        adj = {}
        for i in range(1, 2 * n + 1):
            adj[i] = []
            if i < 2 * n:
                adj[i].append(i + 1)
        
        for u, v in pairs:
            adj[u].append(v)
        
        def bfs(start_node):
            visited = {node: False for node in adj}
            queue = [start_node]
            visited[start_node] = True
            
            while queue:
                node = queue.pop(0)
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            return visited
        
        for i in range(1, 2 * n + 1):
            visited = bfs(i)
            for j in range(1, 2 * n + 1):
                if not visited[j]:
                    return False
        return True

    import itertools
    
    count = 0
    
    
    white_indices = list(range(n))
    
    
    for perm in itertools.permutations(white_indices):
        pairs = []
        valid = True
        for i in range(n):
            pairs.append((whites[perm[i]], blacks[i]))
        
        if is_strongly_connected(pairs):
            count = (count + 1) % 998244353
    
    print(count)

solve()