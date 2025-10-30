class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        from collections import deque
        
        # Function to find the diameter of a tree (given as an edge list)
        def find_diameter(edges: List[List[int]]) -> int:
            n = len(edges) + 1
            adj = [[] for _ in range(n)]
            for a, b in edges:
                adj[a].append(b)
                adj[b].append(a)
            
            # BFS function that returns (farthest_node, distance to farthest_node)
            def bfs(start: int) -> Tuple[int, int]:
                visited = [False]*n
                dist = [0]*n
                visited[start] = True
                queue = deque([start])
                
                while queue:
                    node = queue.popleft()
                    for neigh in adj[node]:
                        if not visited[neigh]:
                            visited[neigh] = True
                            dist[neigh] = dist[node] + 1
                            queue.append(neigh)
                
                # Find the node with maximum distance from 'start'
                far_node = 0
                max_dist = 0
                for i in range(n):
                    if dist[i] > max_dist:
                        max_dist = dist[i]
                        far_node = i
                return far_node, max_dist
            
            # 1) BFS from an arbitrary node (0) to find a farthest node A
            # 2) BFS from A to find a farthest node B, distance(A, B) is diameter
            A, _ = bfs(0)
            B, diameter = bfs(A)
            return diameter
        
        d1 = find_diameter(edges1)
        d2 = find_diameter(edges2)
        
        # Radius of a tree = (diameter + 1) // 2
        r1 = (d1 + 1) // 2
        r2 = (d2 + 1) // 2
        
        # Connecting center nodes yields:
        # new_diameter = max(d1, d2, r1 + r2 + 1)
        return max(d1, d2, r1 + r2 + 1)