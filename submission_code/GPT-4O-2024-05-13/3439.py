from typing import List
from collections import deque

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        
        def tree_diameter(edges: List[List[int]]) -> int:
            def bfs(farthest_node: int) -> (int, int):
                visited = [-1] * len(adj)
                queue = deque([(farthest_node, 0)])
                visited[farthest_node] = 0
                farthest = (farthest_node, 0)
                
                while queue:
                    node, dist = queue.popleft()
                    for neighbor in adj[node]:
                        if visited[neighbor] == -1:
                            visited[neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))
                            if visited[neighbor] > farthest[1]:
                                farthest = (neighbor, visited[neighbor])
                return farthest
            
            n = len(edges) + 1
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            farthest_node = bfs(0)[0]
            diameter = bfs(farthest_node)[1]
            return diameter
        
        diameter1 = tree_diameter(edges1)
        diameter2 = tree_diameter(edges2)
        
        return max(diameter1, diameter2) + 1

# Example usage:
# sol = Solution()
# print(sol.minimumDiameterAfterMerge([[0,1],[0,2],[0,3]], [[0,1]]))  # Output: 3
# print(sol.minimumDiameterAfterMerge([[0,1],[0,2],[0,3],[2,4],[2,5],[3,6],[2,7]], [[0,1],[0,2],[0,3],[2,4],[2,5],[3,6],[2,7]]))  # Output: 5