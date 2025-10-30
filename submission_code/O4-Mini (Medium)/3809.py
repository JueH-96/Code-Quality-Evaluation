from typing import List

class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        n = len(properties)
        # Convert each property list to a set of distinct integers
        prop_sets = [set(prop) for prop in properties]
        
        # Disjoint Set Union (Union-Find) data structure
        parent = list(range(n))
        rank = [0] * n
        
        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            else:
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
        
        # Build edges: if two sets intersect in at least k elements, union their nodes
        for i in range(n):
            for j in range(i + 1, n):
                if len(prop_sets[i].intersection(prop_sets[j])) >= k:
                    union(i, j)
        
        # Count distinct roots = number of connected components
        roots = set(find(i) for i in range(n))
        return len(roots)