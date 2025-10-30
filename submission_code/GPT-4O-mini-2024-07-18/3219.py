from typing import List
from collections import defaultdict

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
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
        
        # Create union-find structure based on the limit
        for i in range(n):
            for j in range(i + 1, n):
                if abs(nums[i] - nums[j]) <= limit:
                    union(i, j)
        
        # Group indices by their root parent
        groups = defaultdict(list)
        for i in range(n):
            root = find(i)
            groups[root].append(i)
        
        # Create the result array
        result = [0] * n
        
        # Sort each group and assign the smallest values
        for indices in groups.values():
            sorted_values = sorted(nums[i] for i in indices)
            indices.sort()
            for idx, value in zip(indices, sorted_values):
                result[idx] = value
        
        return result