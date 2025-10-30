from collections import defaultdict, deque

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def get_diameter(edges):
            if not edges:
                return 0
            
            # Build graph
            n = len(edges) + 1
            graph = defaultdict(list)
            for u, v in edges:
                graph[u].append(v)
                graph[v].append(u)
            
            # Find diameter using two BFS approach
            def bfs(start):
                visited = set()
                queue = deque([(start, 0)])
                visited.add(start)
                farthest_node = start
                farthest_dist = 0
                
                while queue:
                    node, dist = queue.popleft()
                    if dist > farthest_dist:
                        farthest_dist = dist
                        farthest_node = node
                    
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))
                
                return farthest_node, farthest_dist
            
            # First BFS from node 0 to find one end of diameter
            end1, _ = bfs(0)
            # Second BFS from end1 to find actual diameter
            end2, diameter = bfs(end1)
            
            return diameter
        
        d1 = get_diameter(edges1)
        d2 = get_diameter(edges2)
        
        # The minimum diameter after connecting is the max of:
        # 1. Original diameter of tree1
        # 2. Original diameter of tree2  
        # 3. Radius of tree1 + 1 + radius of tree2
        return max(d1, d2, (d1 + 1) // 2 + 1 + (d2 + 1) // 2)