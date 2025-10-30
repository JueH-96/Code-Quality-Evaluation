import math

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        parent = list(range(n))
        
        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]
        
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        for i in range(n):
            for j in range(i + 1, n):
                lcm = (nums[i] * nums[j]) // math.gcd(nums[i], nums[j])
                if lcm <= threshold:
                    union(i, j)

        count = 0
        for i in range(n):
            if parent[i] == i:
                count += 1
        return count