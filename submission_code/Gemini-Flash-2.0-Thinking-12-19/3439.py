from collections import defaultdict, deque

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def get_diameter_and_endpoints(n, edges):
            adj = defaultdict(list)
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)

            def bfs(start_node):
                q = deque([(start_node, 0)])
                visited = {start_node}
                farthest_node = start_node
                max_dist = 0
                while q:
                    u, dist = q.popleft()
                    if dist > max_dist:
                        max_dist = dist
                        farthest_node = u
                    for v in adj[u]:
                        if v not in visited:
                            visited.add(v)
                            q.append((v, dist + 1))
                return farthest_node, max_dist

            if n == 1:
                return 0, [0]

            farthest_node1, _ = bfs(0)
            farthest_node2, diameter = bfs(farthest_node1)
            return diameter, [farthest_node1, farthest_node2]

        n1 = 0
        for u, v in edges1:
            n1 = max(n1, u, v)
        n1 += 1
        n2 = 0
        for u, v in edges2:
            n2 = max(n2, u, v)
        n2 += 1

        if n1 == 0:
            diam2, endpoints2 = get_diameter_and_endpoints(n2, edges2)
            return diam2
        if n2 == 0:
            diam1, endpoints1 = get_diameter_and_endpoints(n1, edges1)
            return diam1


        diam1, endpoints1 = get_diameter_and_endpoints(n1, edges1)
        diam2, endpoints2 = get_diameter_and_endpoints(n2, edges2)

        min_diameter = float('inf')

        for u in endpoints1:
            for v in endpoints2:
                merged_edges = edges1 + [[u, v + n1]] + [[x[0] + n1, x[1] + n1] for x in edges2]
                merged_n = n1 + n2
                current_diameter, _ = get_diameter_and_endpoints(merged_n, merged_edges)
                min_diameter = min(min_diameter, current_diameter)
        
        return min_diameter