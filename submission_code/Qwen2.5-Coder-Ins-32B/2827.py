from typing import List
from math import gcd
from collections import defaultdict

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        if len(nums) == 1:
            return True
        if 1 in nums:
            return False
        
        def prime_factors(n):
            factors = set()
            d = 2
            while d * d <= n:
                while (n % d) == 0:
                    factors.add(d)
                    n //= d
                d += 1
            if n > 1:
                factors.add(n)
            return factors
        
        factor_to_index = defaultdict(list)
        for i, num in enumerate(nums):
            for factor in prime_factors(num):
                factor_to_index[factor].append(i)
        
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
        
        for indices in factor_to_index.values():
            for i in range(1, len(indices)):
                union(indices[i - 1], indices[i])
        
        root = find(0)
        for i in range(1, len(nums)):
            if find(i) != root:
                return False
        
        return True