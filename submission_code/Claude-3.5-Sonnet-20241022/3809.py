class Solution:
    def numberOfComponents(self, properties: List[List[int]], k: int) -> int:
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        def intersect(a, b):
            # Convert arrays to sets and find intersection
            set_a = set(a)
            set_b = set(b)
            return len(set_a.intersection(set_b))
        
        n = len(properties)
        # Initialize parent array for union-find
        parent = list(range(n))
        
        # Build graph by checking all pairs of arrays
        for i in range(n):
            for j in range(i + 1, n):
                # If intersection size >= k, add edge between i and j
                if intersect(properties[i], properties[j]) >= k:
                    union(i, j)
        
        # Count number of unique parents to get number of components
        components = set()
        for i in range(n):
            components.add(find(i))
        
        return len(components)