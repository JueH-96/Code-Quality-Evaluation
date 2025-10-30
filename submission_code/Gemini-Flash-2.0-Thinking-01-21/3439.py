import collections

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def get_diameter_and_max_dist(n, edges):
            adj = collections.defaultdict(list)
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)

            if n <= 1:
                return 0, [0] * n if n > 0 else []

            def bfs(start_node):
                q = collections.deque([(start_node, 0)])
                distances = [-1] * n
                distances[start_node] = 0
                farthest_node = start_node
                max_dist = 0
                while q:
                    u, dist = q.popleft()
                    if dist > max_dist:
                        max_dist = dist
                        farthest_node = u
                    for v in adj[u]:
                        if distances[v] == -1:
                            distances[v] = dist + 1
                            q.append((v, dist + 1))
                return farthest_node, distances

            node1, _ = bfs(0)
            node2, distances_from_node1 = bfs(node1)
            diameter = distances_from_node1[node2]

            max_distances = [0] * n
            for i in range(n):
                max_distances[i] = max(bfs(i)[1])
            return diameter, max_distances

        n = 0
        if edges1:
            n = max(max(edge) for edge in edges1) + 1
        else:
            n = 1 if edges1 == [] and edges2 else 0

        m = 0
        if edges2:
            m = max(max(edge) for edge in edges2) + 1
        else:
            m = 1 if edges2 == [] and edges1 else 0


        diameter1, max_dist1 = get_diameter_and_max_dist(n, edges1)
        diameter2, max_dist2 = get_diameter_and_max_dist(m, edges2)

        min_max_dist1 = min(max_dist1) if max_dist1 else 0
        min_max_dist2 = min(max_dist2) if max_dist2 else 0

        if n == 0:
            return diameter2
        if m == 0:
            return diameter1

        min_diameter = float('inf')
        for u in range(n):
            for v in range(m):
                current_diameter = max(diameter1, diameter2, max_dist1[u] + max_dist2[v] + 1)
                min_diameter = min(min_diameter, current_diameter)

        return min_diameter