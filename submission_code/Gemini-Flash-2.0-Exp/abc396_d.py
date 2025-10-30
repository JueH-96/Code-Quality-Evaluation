def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges.append((u, v, w))

    def find_paths(start, end, graph):
        paths = []
        def dfs(current_node, path, visited):
            if current_node == end:
                paths.append(path)
                return
            
            visited[current_node] = True
            
            for neighbor, weight in graph[current_node]:
                if not visited[neighbor]:
                    dfs(neighbor, path + [(current_node, neighbor, weight)], visited.copy())
        
        graph_adj = {i: [] for i in range(1, n + 1)}
        for u, v, w in edges:
            graph_adj[u].append((v, w))
            graph_adj[v].append((u, w))
            
        dfs(start, [], {i: False for i in range(1, n + 1)})
        return paths

    paths = find_paths(1, n, {i: [] for i in range(1, n + 1)})
    
    min_xor = float('inf')
    for path in paths:
        xor_sum = 0
        for _, _, weight in path:
            xor_sum ^= weight
        min_xor = min(min_xor, xor_sum)

    print(min_xor)

solve()