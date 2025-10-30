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

    def is_good(graph):
        for x, y in forbidden_pairs:
            if has_path(graph, x, y):
                return False
        return True

    def has_path(graph, start, end):
        visited = set()
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node == end:
                return True
            if node in visited:
                continue
            visited.add(node)
            for neighbor in graph.get(node, []):
                queue.append(neighbor)
        return False

    def build_graph(edges_list):
        graph = {}
        for u, v in edges_list:
            if u not in graph:
                graph[u] = []
            if v not in graph:
                graph[v] = []
            graph[u].append(v)
            graph[v].append(u)
        return graph

    for p, q in queries:
        new_edges = edges + [(p, q)]
        graph = build_graph(new_edges)
        if is_good(graph):
            print("Yes")
        else:
            print("No")

solve()