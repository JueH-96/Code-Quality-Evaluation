from math import gcd
from collections import defaultdict
from typing import List

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        if len(nums) == 1:
            return True
        
        # Helper function to find the root of a node in the union-find structure
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        # Helper function to union two nodes in the union-find structure
        def union(x, y):
            rootX = find(x)
            rootY = find(y)
            if rootX != rootY:
                parent[rootX] = rootY
        
        # Initialize the union-find structure
        parent = list(range(len(nums)))
        
        # Create a map from prime factors to indices
        prime_to_indices = defaultdict(list)
        
        # Function to get prime factors of a number
        def prime_factors(n):
            i = 2
            factors = set()
            while i * i <= n:
                while (n % i) == 0:
                    factors.add(i)
                    n //= i
                i += 1
            if n > 1:
                factors.add(n)
            return factors
        
        # Populate the prime_to_indices map
        for i, num in enumerate(nums):
            factors = prime_factors(num)
            for factor in factors:
                prime_to_indices[factor].append(i)
        
        # Union all indices that share a common prime factor
        for indices in prime_to_indices.values():
            for i in range(1, len(indices)):
                union(indices[0], indices[i])
        
        # Check if all indices are connected
        root = find(0)
        for i in range(1, len(nums)):
            if find(i) != root:
                return False
        
        return True