class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        import math
        
        class UnionFind:
            def __init__(self, n):
                self.parent = list(range(n))
                self.components = n
            
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                px, py = self.find(x), self.find(y)
                if px != py:
                    self.parent[py] = px
                    self.components -= 1
        
        n = len(nums)
        uf = UnionFind(n)
        
        for i in range(n):
            if nums[i] > threshold:
                continue
            for j in range(i + 1, n):
                if nums[j] > threshold:
                    continue
                lcm_val = nums[i] * nums[j] // math.gcd(nums[i], nums[j])
                if lcm_val <= threshold:
                    uf.union(i, j)
        
        return uf.components