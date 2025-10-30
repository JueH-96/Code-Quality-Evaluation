from collections import deque

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        n = len(edges1) + 1
        m = len(edges2) + 1

        adj1 = [[] for _ in range(n)]
        for u, v in edges1:
            adj1[u].append(v)
            adj1[v].append(u)

        adj2 = [[] for _ in range(m)]
        for u, v in edges2:
            adj2[u].append(v)
            adj2[v].append(u)

        def get_diameter(adj):
            num_nodes = len(adj)
            if num_nodes <= 1:
                return 0

            def bfs(start_node):
                distances = [-1] * num_nodes
                distances[start_node] = 0
                queue = deque([start_node])
                farthest_node = start_node
                max_distance = 0

                while queue:
                    u = queue.popleft()
                    if distances[u] > max_distance:
                        max_distance = distances[u]
                        farthest_node = u

                    for v in adj[u]:
                        if distances[v] == -1:
                            distances[v] = distances[u] + 1
                            queue.append(v)
                return farthest_node, max_distance

            farthest1, _ = bfs(0)
            farthest2, diameter = bfs(farthest1)
            return diameter

        def get_max_distance(start_node, adj):
            num_nodes = len(adj)
            distances = [-1] * num_nodes
            distances[start_node] = 0
            queue = deque([start_node])
            max_distance = 0

            while queue:
                u = queue.popleft()
                max_distance = max(max_distance, distances[u])
                for v in adj[u]:
                    if distances[v] == -1:
                        distances[v] = distances[u] + 1
                        queue.append(v)
            return max_distance

        diameter1 = get_diameter(adj1)
        diameter2 = get_diameter(adj2)
        min_diameter = float('inf')

        max_dist1 = [0] * n
        for i in range(n):
            max_dist1[i] = get_max_distance(i, adj1)

        max_dist2 = [0] * m
        for i in range(m):
            max_dist2[i] = get_max_distance(i, adj2)

        for i in range(n):
            for j in range(m):
                min_diameter = min(min_diameter, max(diameter1, diameter2, max_dist1[i] + 1 + max_dist2[j]))

        return min_diameter