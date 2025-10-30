from typing import List
from math import gcd
from collections import defaultdict

class Solution:
    def countComponents(self, nums: List[int], threshold: int) -> int:
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            rootX = find(x)
            rootY = find(y)
            if rootX != rootY:
                parent[rootX] = rootY
        
        def lcm(x, y):
            return x * y // gcd(x, y)
        
        n = len(nums)
        parent = list(range(n))
        
        # Map each number to its index
        num_to_index = {num: i for i, num in enumerate(nums)}
        
        # Connect nodes
        for i in range(n):
            for j in range(i + 1, n):
                if lcm(nums[i], nums[j]) <= threshold:
                    union(i, j)
        
        # Count unique roots which represent unique connected components
        unique_components = len(set(find(i) for i in range(n)))
        return unique_components

# Example usage
sol = Solution()
print(sol.countComponents([2,4,8,3,9], 5))  # Output: 4
print(sol.countComponents([2,4,8,3,9,12], 10))  # Output: 2