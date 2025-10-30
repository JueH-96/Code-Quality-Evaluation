from collections import defaultdict
from heapq import heappush, heappop

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def build_tree(edges):
            tree = defaultdict(list)
            for u, v in edges:
                tree[u].append(v)
                tree[v].append(u)
            return tree
        
        def find_furthest(node, tree):
            visited = set([node])
            stack = [(0, node)]
            max_distance, max_node = 0, node
            while stack:
                distance, current = heappop(stack)
                if distance > max_distance:
                    max_distance = distance
                    max_node = current
                for neighbor in tree[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        heappush(stack, (distance + 1, neighbor))
            return max_node
        
        def find_diameter(node, tree):
            visited = set([node])
            stack = [(0, node)]
            max_distance = 0
            while stack:
                distance, current = heappop(stack)
                max_distance = max(max_distance, distance)
                for neighbor in tree[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        heappush(stack, (distance + 1, neighbor))
            return max_distance
        
        tree1 = build_tree(edges1)
        tree2 = build_tree(edges2)
        
        root1 = find_furthest(0, tree1)
        diameter1 = find_diameter(root1, tree1)
        
        root2 = find_furthest(0, tree2)
        diameter2 = find_diameter(root2, tree2)
        
        return max(diameter1, diameter2) + 1