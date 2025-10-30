class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        def lcm(a, b):
            def gcd(x, y):
                while y:
                    x, y = y, x % y
                return x
            return abs(a * b) // gcd(a, b)
        
        n = len(nums)
        parent = list(range(n))
        
        # Check all pairs of nodes
        for i in range(n):
            for j in range(i + 1, n):
                # If LCM is less than or equal to threshold, connect the nodes
                if lcm(nums[i], nums[j]) <= threshold:
                    union(i, j)
        
        # Count unique parents to get number of components
        components = set()
        for i in range(n):
            components.add(find(i))
        
        return len(components)