def solve():
    n, p = map(int, input().split())
    
    def count_connected_graphs(n, m, p):
        if m < n - 1 or m > n * (n - 1) // 2:
            return 0
        
        def is_connected(adj_matrix):
            visited = [False] * n
            q = [0]
            visited[0] = True
            count = 0
            
            while q:
                u = q.pop(0)
                count += 1
                for v in range(n):
                    if adj_matrix[u][v] == 1 and not visited[v]:
                        visited[v] = True
                        q.append(v)
            
            return count == n
        
        def bfs_distance(adj_matrix):
            distances = [-1] * n
            distances[0] = 0
            q = [0]
            
            while q:
                u = q.pop(0)
                for v in range(n):
                    if adj_matrix[u][v] == 1 and distances[v] == -1:
                        distances[v] = distances[u] + 1
                        q.append(v)
            return distances
        
        def check_condition(adj_matrix):
            distances = bfs_distance(adj_matrix)
            even_count = 0
            odd_count = 0
            for dist in distances:
                if dist % 2 == 0:
                    even_count += 1
                else:
                    odd_count += 1
            return even_count == odd_count
        
        count = 0
        
        import itertools
        
        edges = list(itertools.combinations(range(n), 2))
        
        for indices in itertools.combinations(range(len(edges)), m):
            adj_matrix = [[0] * n for _ in range(n)]
            
            for index in indices:
                u, v = edges[index]
                adj_matrix[u][v] = 1
                adj_matrix[v][u] = 1
            
            if is_connected(adj_matrix) and check_condition(adj_matrix):
                count = (count + 1) % p
        
        return count
    
    results = []
    for m in range(n - 1, n * (n - 1) // 2 + 1):
        results.append(str(count_connected_graphs(n, m, p)))
    
    print(" ".join(results))

solve()