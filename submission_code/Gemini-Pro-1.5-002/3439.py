class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def bfs(graph, start_node):
            max_dist = 0
            farthest_node = start_node
            queue = [(start_node, 0)]
            visited = {start_node}
            while queue:
                curr_node, dist = queue.pop(0)
                if dist > max_dist:
                    max_dist = dist
                    farthest_node = curr_node
                for neighbor in graph[curr_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            return farthest_node, max_dist

        def get_diameter(edges):
            if not edges:
                return 0
            n = len(edges) + 1
            graph = {i: [] for i in range(n)}
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)
            
            farthest1, _ = bfs(graph, 0)
            farthest2, max_dist = bfs(graph, farthest1)
            return max_dist
        
        def get_max_depth(edges, start_node):
            if not edges:
                return 0

            n = len(edges) + 1
            graph = {i: [] for i in range(n)}
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)

            max_depth = 0
            queue = [(start_node, 0)]
            visited = {start_node}

            while queue:
                curr_node, depth = queue.pop(0)
                max_depth = max(max_depth, depth)
                for neighbor in graph[curr_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
            return max_depth


        n = len(edges1) + 1
        m = len(edges2) + 1

        diameter1 = get_diameter(edges1)
        diameter2 = get_diameter(edges2)

        max_depth1 = 0
        max_depth2 = 0

        if edges1:
            farthest1_1, _ = bfs({i: [] for i in range(n)}, 0)
            farthest1_2, _ = bfs({i: [] for i in range(n)}, farthest1_1)
            max_depth1 = max(get_max_depth(edges1, farthest1_1), get_max_depth(edges1, farthest1_2))

        if edges2:
            farthest2_1, _ = bfs({i: [] for i in range(m)}, 0)
            farthest2_2, _ = bfs({i: [] for i in range(m)}, farthest2_1)
            max_depth2 = max(get_max_depth(edges2, farthest2_1), get_max_depth(edges2, farthest2_2))



        return max(diameter1, diameter2, max_depth1 + max_depth2 + 1)