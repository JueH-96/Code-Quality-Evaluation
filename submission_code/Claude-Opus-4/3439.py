class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        from collections import defaultdict, deque
        
        def build_graph(edges, n):
            graph = defaultdict(list)
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)
            return graph
        
        def bfs_farthest(graph, start, n):
            visited = [False] * n
            dist = [0] * n
            queue = deque([start])
            visited[start] = True
            
            farthest_node = start
            max_dist = 0
            
            while queue:
                node = queue.popleft()
                for neighbor in graph[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        dist[neighbor] = dist[node] + 1
                        queue.append(neighbor)
                        if dist[neighbor] > max_dist:
                            max_dist = dist[neighbor]
                            farthest_node = neighbor
            
            return farthest_node, max_dist
        
        def find_diameter_and_center(edges, n):
            if n == 1:
                return 0, 0
            
            graph = build_graph(edges, n)
            
            # Find one end of diameter
            end1, _ = bfs_farthest(graph, 0, n)
            
            # Find other end of diameter and the diameter length
            end2, diameter = bfs_farthest(graph, end1, n)
            
            # Find the center(s) of the tree
            # The center minimizes the maximum distance to any other node
            # For a tree, this is the middle node(s) of the diameter path
            
            # Find the path from end1 to end2
            parent = [-1] * n
            visited = [False] * n
            queue = deque([end1])
            visited[end1] = True
            
            while queue:
                node = queue.popleft()
                if node == end2:
                    break
                for neighbor in graph[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        parent[neighbor] = node
                        queue.append(neighbor)
            
            # Reconstruct the diameter path
            path = []
            current = end2
            while current != -1:
                path.append(current)
                current = parent[current]
            
            # The radius is the maximum distance from center to any node
            # For optimal connection, we want the node that minimizes this
            center_idx = len(path) // 2
            center = path[center_idx]
            
            # Calculate radius (max distance from center to any node)
            _, radius = bfs_farthest(graph, center, n)
            
            return diameter, radius
        
        n1 = len(edges1) + 1
        n2 = len(edges2) + 1
        
        diameter1, radius1 = find_diameter_and_center(edges1, n1)
        diameter2, radius2 = find_diameter_and_center(edges2, n2)
        
        # The new diameter is the maximum of:
        # 1. Original diameter of tree1
        # 2. Original diameter of tree2
        # 3. The longest path through the connecting edge (radius1 + 1 + radius2)
        
        return max(diameter1, diameter2, radius1 + radius2 + 1)