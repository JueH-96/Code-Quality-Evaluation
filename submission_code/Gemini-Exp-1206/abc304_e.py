def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v = map(int, input().split())
        edges.append((u, v))
    
    k = int(input())
    forbidden_pairs = []
    for _ in range(k):
        x, y = map(int, input().split())
        forbidden_pairs.append((x, y))
    
    q = int(input())
    queries = []
    for _ in range(q):
        p, q = map(int, input().split())
        queries.append((p, q))
    
    def bfs(graph, start, end):
        if start == end:
            return True
        
        queue = [(start, [start])]
        visited = {start}
        
        while queue:
            curr, path = queue.pop(0)
            
            if curr == end:
                return True
            
            for neighbor in graph.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return False

    for p, q in queries:
        graph = {}
        for u, v in edges:
            graph.setdefault(u, []).append(v)
            graph.setdefault(v, []).append(u)
        
        graph.setdefault(p, []).append(q)
        graph.setdefault(q, []).append(p)
        
        is_good = True
        for x, y in forbidden_pairs:
            if bfs(graph, x, y):
                is_good = False
                break
        
        print("Yes" if is_good else "No")

solve()