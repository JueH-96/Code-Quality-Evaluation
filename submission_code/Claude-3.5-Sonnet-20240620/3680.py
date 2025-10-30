class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            root_x, root_y = find(x), find(y)
            if root_x != root_y:
                parent[root_x] = root_y
        
        def lcm(a, b):
            return abs(a * b) // math.gcd(a, b)
        
        n = len(nums)
        parent = list(range(n))
        
        for i in range(n):
            for j in range(i + 1, n):
                if lcm(nums[i], nums[j]) <= threshold:
                    union(i, j)
        
        components = set()
        for i in range(n):
            components.add(find(i))
        
        return len(components)