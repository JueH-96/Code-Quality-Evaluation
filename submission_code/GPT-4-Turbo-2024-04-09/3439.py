from typing import List
from collections import deque, defaultdict

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def build_tree(edges):
            tree = defaultdict(list)
            for u, v in edges:
                tree[u].append(v)
                tree[v].append(u)
            return tree
        
        def bfs(tree, start_node):
            queue = deque([start_node])
            distances = {start_node: 0}
            farthest_node = start_node
            max_distance = 0
            
            while queue:
                node = queue.popleft()
                current_distance = distances[node]
                
                for neighbor in tree[node]:
                    if neighbor not in distances:
                        distances[neighbor] = current_distance + 1
                        queue.append(neighbor)
                        if distances[neighbor] > max_distance:
                            max_distance = distances[neighbor]
                            farthest_node = neighbor
                            
            return farthest_node, max_distance, distances
        
        def find_diameter(tree):
            # Find the farthest node from an arbitrary node (here 0)
            farthest, _, _ = bfs(tree, 0)
            # Use the farthest node found to find the actual diameter
            _, diameter, distances = bfs(tree, farthest)
            return diameter, distances
        
        # Build trees from edges
        tree1 = build_tree(edges1)
        tree2 = build_tree(edges2)
        
        # Find diameters and distances from any node (choosing 0) to all other nodes
        diameter1, distances1 = find_diameter(tree1)
        diameter2, distances2 = find_diameter(tree2)
        
        # Find the node that maximizes the distance from node 0 in each tree
        max_dist1 = max(distances1.values())
        max_dist2 = max(distances2.values())
        
        # Calculate the minimum possible diameter after merging
        # We consider connecting the nodes that are farthest from the root in each tree
        # The new diameter is the maximum of the three possible values:
        # 1. Diameter of the first tree
        # 2. Diameter of the second tree
        # 3. Longest path from root of tree1 to root of tree2 through the new edge
        min_possible_diameter = min(max(diameter1, diameter2, max_dist1 + max_dist2 + 1))
        
        return min_possible_diameter

# Example usage:
sol = Solution()
print(sol.minimumDiameterAfterMerge([[0,1],[0,2],[0,3]], [[0,1]]))  # Output: 3
print(sol.minimumDiameterAfterMerge([[0,1],[0,2],[0,3],[2,4],[2,5],[3,6],[2,7]], [[0,1],[0,2],[0,3],[2,4],[2,5],[3,6],[2,7]]))  # Output: 5