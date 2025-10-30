from math import isqrt
from collections import defaultdict

class Solution:
    def canTraverseAllPairs(self, nums: list[int]) -> bool:
        n = len(nums)
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[py] = px
        
        # Function to compute all prime factors of a number
        def prime_factors(x: int) -> set[int]:
            factors = set()
            # Check 2 separately
            while x % 2 == 0:
                factors.add(2)
                x //= 2
            # Check odd factors
            f = 3
            while f <= isqrt(x) and x > 1:
                if x % f == 0:
                    factors.add(f)
                    x //= f
                else:
                    f += 2
            if x > 1:
                factors.add(x)
            return factors
        
        # map prime factor to one of indices having that factor
        factor_to_index = {}
        
        for i, num in enumerate(nums):
            factors = prime_factors(num)
            for f in factors:
                if f in factor_to_index:
                    # union current index with previous index having factor f
                    union(i, factor_to_index[f])
                else:
                    factor_to_index[f] = i
        
        # check if all indices belong to the same component
        root = find(0)
        for i in range(1, n):
            if find(i) != root:
                return False
        return True