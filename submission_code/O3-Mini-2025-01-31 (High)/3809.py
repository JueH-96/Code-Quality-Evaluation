from typing import List

class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        n = len(properties)
        # Union-Find initialization: each node is its own parent.
        parent = list(range(n))
        
        def find(x):
            # Path compression for efficient find
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            rootX = find(x)
            rootY = find(y)
            if rootX != rootY:
                parent[rootY] = rootX
        
        # Convert each list in properties to a set to remove duplicates,
        # because we need the count of distinct common elements.
        prop_sets = [set(prop) for prop in properties]
        
        # Build the graph edges. Check every pair (i, j) and add an edge if
        # the intersection of their sets has at least k elements.
        for i in range(n):
            for j in range(i + 1, n):
                if len(prop_sets[i].intersection(prop_sets[j])) >= k:
                    union(i, j)
        
        # Count the number of unique connected components by counting unique roots.
        components = len({find(i) for i in range(n)})
        return components