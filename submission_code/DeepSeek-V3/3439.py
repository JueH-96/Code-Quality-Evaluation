from collections import deque
from typing import List

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def build_tree(edges, n):
            tree = [[] for _ in range(n)]
            for u, v in edges:
                tree[u].append(v)
                tree[v].append(u)
            return tree
        
        def find_farthest_node(tree, start):
            visited = [False] * len(tree)
            queue = deque()
            queue.append((start, 0))
            visited[start] = True
            farthest_node = start
            max_distance = 0
            while queue:
                node, distance = queue.popleft()
                if distance > max_distance:
                    max_distance = distance
                    farthest_node = node
                for neighbor in tree[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append((neighbor, distance + 1))
            return farthest_node, max_distance
        
        def find_diameter(tree):
            n = len(tree)
            if n == 0:
                return 0
            start = 0
            farthest, _ = find_farthest_node(tree, start)
            _, diameter = find_farthest_node(tree, farthest)
            return diameter
        
        def find_centers(tree, diameter):
            n = len(tree)
            if n == 0:
                return []
            start = 0
            farthest, _ = find_farthest_node(tree, start)
            farthest2, _ = find_farthest_node(tree, farthest)
            path = []
            visited = [False] * n
            def dfs(node, target):
                if node == target:
                    path.append(node)
                    return True
                visited[node] = True
                for neighbor in tree[node]:
                    if not visited[neighbor]:
                        if dfs(neighbor, target):
                            path.append(node)
                            return True
                return False
            dfs(farthest, farthest2)
            if diameter % 2 == 0:
                return [path[diameter // 2]]
            else:
                return [path[diameter // 2], path[diameter // 2 + 1]]
        
        n = len(edges1) + 1
        m = len(edges2) + 1
        tree1 = build_tree(edges1, n)
        tree2 = build_tree(edges2, m)
        
        diameter1 = find_diameter(tree1)
        diameter2 = find_diameter(tree2)
        
        centers1 = find_centers(tree1, diameter1)
        centers2 = find_centers(tree2, diameter2)
        
        min_diameter = float('inf')
        for c1 in centers1:
            for c2 in centers2:
                new_diameter = max(diameter1, diameter2, (diameter1 + 1) // 2 + (diameter2 + 1) // 2 + 1)
                if new_diameter < min_diameter:
                    min_diameter = new_diameter
        return min_diameter