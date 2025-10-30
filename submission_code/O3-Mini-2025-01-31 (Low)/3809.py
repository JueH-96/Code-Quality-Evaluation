class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        n = len(properties)
        
        # Precompute the distinct values in each property list
        sets = [set(lst) for lst in properties]
        
        # DSU helper functions
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            rootX = find(x)
            rootY = find(y)
            if rootX != rootY:
                parent[rootY] = rootX
        
        # Loop over all pairs to decide if there should be an edge
        for i in range(n):
            for j in range(i+1, n):
                if len(sets[i].intersection(sets[j])) >= k:
                    union(i, j)
        
        # Count distinct connected components
        comp_roots = set(find(i) for i in range(n))
        return len(comp_roots)