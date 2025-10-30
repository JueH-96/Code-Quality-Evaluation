import itertools

def solve():
    n = int(input())
    edges = []
    for _ in range(n - 1):
        u, v, l = map(int, input().split())
        edges.append((u, v, l))

    graph = {i: [] for i in range(1, n + 1)}
    for u, v, l in edges:
        graph[u].append((v, l))
        graph[v].append((u, l))

    def dijkstra(start, end, avoid):
        dist = {i: float('inf') for i in range(1, n + 1)}
        dist[start] = 0
        visited = set()
        pq = [(0, start)]

        while pq:
            d, u = min(pq)
            if u == end:
                return d
            if u in visited:
                continue
            visited.add(u)
            for v, l in graph[u]:
                if v not in avoid:
                    if dist[v] > dist[u] + l:
                        dist[v] = dist[u] + l
                        pq.append((dist[v], v))
            pq.sort()
        return float('inf')

    def calculate_score(selected_nodes):
        
        total_dist = 0
        
        
        path = [1]
        
        
        unvisited = set(selected_nodes)
        
        
        current_node = 1
        
        while unvisited:
            next_node = min(unvisited, key=lambda node: dijkstra(current_node, node, set()))
            
            total_dist += dijkstra(current_node, next_node, set())
            
            current_node = next_node
            unvisited.remove(next_node)
            path.append(next_node)

        total_dist += dijkstra(current_node, 1, set())
        return total_dist

    for k in range(1, n + 1):
        max_score = 0
        for selected_nodes in itertools.combinations(range(1, n + 1), k):
            
            if 1 not in selected_nodes:
                score = calculate_score(list(selected_nodes))
                max_score = max(max_score, score)
            else:
                
                score = calculate_score(list(selected_nodes))
                max_score = max(max_score, score)

        print(max_score)

solve()