from typing import List
from collections import defaultdict

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def build_tree(edges):
            tree = defaultdict(list)
            for u, v in edges:
                tree[u].append(v)
                tree[v].append(u)
            return tree
        
        def bfs(tree, start):
            visited = set()
            queue = [(start, 0)]
            farthest_node = start
            max_distance = 0
            while queue:
                node, distance = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                if distance > max_distance:
                    max_distance = distance
                    farthest_node = node
                for neighbor in tree[node]:
                    if neighbor not in visited:
                        queue.append((neighbor, distance + 1))
            return farthest_node, max_distance
        
        def tree_diameter(edges):
            tree = build_tree(edges)
            # Find the farthest node from any node (let's take node 0)
            farthest_node, _ = bfs(tree, 0)
            # Find the farthest node from the farthest node found in the previous step
            _, diameter = bfs(tree, farthest_node)
            return diameter
        
        # Calculate the diameters of both trees
        diameter1 = tree_diameter(edges1)
        diameter2 = tree_diameter(edges2)
        
        # The minimum possible diameter after connecting the trees is the maximum of the two diameters
        # plus 1 (for the new edge connecting the trees)
        return max(diameter1, diameter2) + 1

# Example usage:
# sol = Solution()
# print(sol.minimumDiameterAfterMerge([[0,1],[0,2],[0,3]], [[0,1]])) # Output: 3
# print(sol.minimumDiameterAfterMerge([[0,1],[0,2],[0,3],[2,4],[2,5],[3,6],[2,7]], [[0,1],[0,2],[0,3],[2,4],[2,5],[3,6],[2,7]])) # Output: 5