from typing import List
from math import gcd
from collections import defaultdict

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
                parent[rootY] = rootX
        
        # Initialize union-find structure
        n = len(nums)
        parent = list(range(n))
        
        # Map each prime factor to the indices of nums that have this prime factor
        prime_to_indices = defaultdict(list)
        
        # Function to get prime factors of a number
        def get_prime_factors(x):
            factors = set()
            d = 2
            while d * d <= x:
                while x % d == 0:
                    factors.add(d)
                    x //= d
                d += 1
            if x > 1:
                factors.add(x)
            return factors
        
        # Populate the prime_to_indices map
        for i, num in enumerate(nums):
            prime_factors = get_prime_factors(num)
            for prime in prime_factors:
                prime_to_indices[prime].append(i)
        
        # Union indices that share a common prime factor
        for indices in prime_to_indices.values():
            for i in range(1, len(indices)):
                union(indices[0], indices[i])
        
        # Check if all indices are connected
        root = find(0)
        for i in range(1, n):
            if find(i) != root:
                return False
        
        return True