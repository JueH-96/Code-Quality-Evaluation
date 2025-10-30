class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        from collections import defaultdict, deque
        import math
        
        def build_graph(edges, n):
            graph = defaultdict(list)
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)
            return graph
        
        def bfs_farthest(graph, start, n):
            visited = [False] * n
            queue = deque([(start, 0)])
            visited[start] = True
            farthest_node = start
            max_dist = 0
            
            while queue:
                node, dist = queue.popleft()
                if dist > max_dist:
                    max_dist = dist
                    farthest_node = node
                
                for neighbor in graph[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append((neighbor, dist + 1))
            
            return farthest_node, max_dist
        
        def find_diameter(edges, n):
            if n == 1:
                return 0
            
            graph = build_graph(edges, n)
            
            # Find one end of the diameter
            farthest1, _ = bfs_farthest(graph, 0, n)
            
            # Find the other end and the diameter
            farthest2, diameter = bfs_farthest(graph, farthest1, n)
            
            return diameter
        
        n1 = len(edges1) + 1
        n2 = len(edges2) + 1
        
        diameter1 = find_diameter(edges1, n1)
        diameter2 = find_diameter(edges2, n2)
        
        # The radius is ceil(diameter/2)
        radius1 = math.ceil(diameter1 / 2)
        radius2 = math.ceil(diameter2 / 2)
        
        # The minimum diameter after connecting is the maximum of:
        # 1. Original diameter of tree 1
        # 2. Original diameter of tree 2
        # 3. radius1 + 1 + radius2 (path through the connecting edge)
        return max(diameter1, diameter2, radius1 + 1 + radius2)