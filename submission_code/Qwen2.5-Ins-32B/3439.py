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
        
        def dfs(node, parent, depth, tree, depths):
            depths[node] = depth
            for neighbor in tree[node]:
                if neighbor != parent:
                    dfs(neighbor, node, depth + 1, tree, depths)
        
        def find_farthest_node(tree, start):
            depths = {}
            dfs(start, -1, 0, tree, depths)
            farthest_node = max(depths, key=depths.get)
            return farthest_node, depths[farthest_node]
        
        tree1 = build_tree(edges1)
        tree2 = build_tree(edges2)
        
        # Find the farthest node from an arbitrary node in each tree
        farthest_node1, _ = find_farthest_node(tree1, 0)
        farthest_node2, _ = find_farthest_node(tree2, 0)
        
        # Find the diameter of each tree
        _, diameter1 = find_farthest_node(tree1, farthest_node1)
        _, diameter2 = find_farthest_node(tree2, farthest_node2)
        
        # Calculate the minimum possible diameter after merging
        min_diameter = float('inf')
        for node1 in range(len(edges1) + 1):
            for node2 in range(len(edges2) + 1):
                depths1 = {}
                depths2 = {}
                dfs(node1, -1, 0, tree1, depths1)
                dfs(node2, -1, 0, tree2, depths2)
                max_depth1 = max(depths1.values())
                max_depth2 = max(depths2.values())
                min_diameter = min(min_diameter, max(diameter1, diameter2, max_depth1 + max_depth2 + 1))
        
        return min_diameter