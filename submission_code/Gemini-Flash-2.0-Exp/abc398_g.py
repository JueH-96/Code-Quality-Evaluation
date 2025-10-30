def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v = map(int, input().split())
        edges.append((u, v))

    def is_bipartite(graph):
        colors = {}
        for node in graph:
            if node not in colors:
                if not bfs(graph, node, colors):
                    return False
        return True

    def bfs(graph, start_node, colors):
        queue = [(start_node, 0)]  # 0 represents color 0, 1 represents color 1
        colors[start_node] = 0

        while queue:
            node, color = queue.pop(0)

            for neighbor in graph.get(node, []):
                if neighbor not in colors:
                    colors[neighbor] = 1 - color
                    queue.append((neighbor, 1 - color))
                elif colors[neighbor] == color:
                    return False
        return True

    def can_add_edge(graph, u, v):
        if u in graph and v in graph[u]:
            return False

        new_graph = {}
        for node in graph:
            new_graph[node] = set(graph[node])
        
        if u not in new_graph:
            new_graph[u] = set()
        if v not in new_graph:
            new_graph[v] = set()
            
        new_graph[u].add(v)
        new_graph[v].add(u)
        
        return is_bipartite(new_graph)

    def get_possible_moves(graph, n):
        moves = []
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                if can_add_edge(graph, i, j):
                    moves.append((i, j))
        return moves

    def build_graph(n, edges):
        graph = {}
        for u, v in edges:
            if u not in graph:
                graph[u] = set()
            if v not in graph:
                graph[v] = set()
            graph[u].add(v)
            graph[v].add(u)
        return graph

    graph = build_graph(n, edges)
    
    def calculate_grundy(graph, n, memo):
        graph_tuple = tuple(sorted((u, tuple(sorted(v))) for u, v in graph.items()))
        if graph_tuple in memo:
            return memo[graph_tuple]

        possible_moves = get_possible_moves(graph, n)
        mex_set = set()

        for u, v in possible_moves:
            new_graph = {}
            for node in graph:
                new_graph[node] = set(graph[node])
            
            if u not in new_graph:
                new_graph[u] = set()
            if v not in new_graph:
                new_graph[v] = set()
                
            new_graph[u].add(v)
            new_graph[v].add(u)
            
            mex_set.add(calculate_grundy(new_graph, n, memo))

        grundy = 0
        while grundy in mex_set:
            grundy += 1

        memo[graph_tuple] = grundy
        return grundy
    
    memo = {}
    grundy_value = calculate_grundy(graph, n, memo)

    if grundy_value > 0:
        print("Aoki")
    else:
        print("Takahashi")

solve()