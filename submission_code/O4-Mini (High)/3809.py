from typing import List

class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        n = len(properties)
        # Disjoint Set Union (Union-Find) setup
        parent = list(range(n))
        
        def find(x: int) -> int:
            # path compression
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        
        # Convert each property list to a set of distinct integers
        prop_sets = [set(arr) for arr in properties]
        
        # For each pair, if they share at least k distinct values, union them
        for i in range(n):
            for j in range(i + 1, n):
                if len(prop_sets[i].intersection(prop_sets[j])) >= k:
                    union(i, j)
        
        # Count distinct roots = number of connected components
        roots = set(find(i) for i in range(n))
        return len(roots)