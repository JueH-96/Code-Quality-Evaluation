from math import gcd
from collections import defaultdict

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        if len(nums) == 1:
            return True
        
        # Find the maximum number in the list
        max_num = max(nums)
        
        # If the maximum number is 1, we cannot traverse between any indices
        if max_num == 1:
            return False
        
        # Create a union-find data structure
        parent = list(range(len(nums)))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            rootX = find(x)
            rootY = find(y)
            if rootX != rootY:
                parent[rootX] = rootY
        
        # Group numbers that have a common factor greater than 1
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if gcd(nums[i], nums[j]) > 1:
                    union(i, j)
        
        # Check if all numbers are in the same group
        root = find(0)
        for i in range(1, len(nums)):
            if find(i) != root:
                return False
        
        return True