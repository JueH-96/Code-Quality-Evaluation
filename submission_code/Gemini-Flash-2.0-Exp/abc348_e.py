def solve():
    n = int(input())
    edges = []
    for _ in range(n - 1):
        u, v = map(int, input().split())
        edges.append((u - 1, v - 1))
    c = list(map(int, input().split()))

    def calculate_f(x):
        f_x = 0
        for i in range(n):
            distance = bfs(x, i, edges, n)
            f_x += c[i] * distance
        return f_x

    def bfs(start, end, edges, n):
        if start == end:
            return 0
        
        graph = [[] for _ in range(n)]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        queue = [(start, 0)]
        visited = {start}
        
        while queue:
            node, dist = queue.pop(0)
            
            if node == end:
                return dist
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return float('inf')

    min_f = float('inf')
    for v in range(n):
        min_f = min(min_f, calculate_f(v))

    print(min_f)

solve()