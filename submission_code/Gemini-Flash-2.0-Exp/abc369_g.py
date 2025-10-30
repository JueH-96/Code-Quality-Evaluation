def solve():
    n = int(input())
    edges = []
    for _ in range(n - 1):
        u, v, l = map(int, input().split())
        edges.append((u, v, l))

    def calculate_score(k, chosen_vertices):
        """Calculates the minimum score Takahashi can achieve given Aoki's choice."""
        
        import heapq
        
        def dijkstra(start, end):
            """Finds the shortest path between start and end."""
            dist = {node: float('inf') for node in range(1, n + 1)}
            dist[start] = 0
            pq = [(0, start)]

            while pq:
                d, u = heapq.heappop(pq)

                if d > dist[u]:
                    continue

                for v, l in adj[u]:
                    if dist[v] > dist[u] + l:
                        dist[v] = dist[u] + l
                        heapq.heappush(pq, (dist[v], v))
            return dist[end]

        adj = {i: [] for i in range(1, n + 1)}
        for u, v, l in edges:
            adj[u].append((v, l))
            adj[v].append((u, l))

        # Calculate shortest paths from 1 to each chosen vertex
        distances_from_1 = {v: dijkstra(1, v) for v in chosen_vertices}

        # Calculate all pairwise shortest paths between chosen vertices
        pairwise_distances = {}
        for u in chosen_vertices:
            pairwise_distances[u] = {}
            for v in chosen_vertices:
                pairwise_distances[u][v] = dijkstra(u, v)

        import itertools
        
        min_walk_length = float('inf')
        for permutation in itertools.permutations(chosen_vertices):
            walk_length = distances_from_1[permutation[0]]
            for i in range(len(permutation) - 1):
                walk_length += pairwise_distances[permutation[i]][permutation[i+1]]
            walk_length += distances_from_1[permutation[-1]]
            min_walk_length = min(min_walk_length, walk_length)

        return min_walk_length

    for k in range(1, n + 1):
        import itertools
        
        max_min_score = 0
        for chosen_vertices in itertools.combinations(range(1, n + 1), k):
            max_min_score = max(max_min_score, calculate_score(k, chosen_vertices))
        print(max_min_score)

solve()