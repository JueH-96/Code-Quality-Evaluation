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
        queue = [start]
        visited = {start}
        while queue:
            curr = queue.pop(0)
            if curr == end:
                return True
            for neighbor in graph.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    def is_good(graph, forbidden_pairs):
        for x, y in forbidden_pairs:
            if bfs(graph, x, y):
                return False
        return True

    for p, q in queries:
        temp_edges = edges + [(p, q)]
        graph = {}
        for u, v in temp_edges:
            graph.setdefault(u, []).append(v)
            graph.setdefault(v, []).append(u)
        
        if is_good(graph, forbidden_pairs):
            print("Yes")
        else:
            print("No")

solve()