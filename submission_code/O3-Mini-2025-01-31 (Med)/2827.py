from typing import List
import math

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
        
    def find(self, a):
        if self.parent[a] != a:
            self.parent[a] = self.find(self.parent[a])
        return self.parent[a]
    
    def union(self, a, b):
        pa = self.find(a)
        pb = self.find(b)
        if pa == pb:
            return
        if self.rank[pa] < self.rank[pb]:
            self.parent[pa] = pb
        elif self.rank[pa] > self.rank[pb]:
            self.parent[pb] = pa
        else:
            self.parent[pb] = pa
            self.rank[pa] += 1

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        if n == 1:
            return True
        
        dsu = DSU(n)
        # mapping factor -> index where it was first seen
        factor_to_index = {}
        
        # Function to get unique prime factors for a given number
        def get_prime_factors(num):
            factors = set()
            # Check divisibility by 2
            while num % 2 == 0:
                factors.add(2)
                num //= 2
            # Check odd factors
            factor = 3
            while factor * factor <= num:
                if num % factor == 0:
                    factors.add(factor)
                    while num % factor == 0:
                        num //= factor
                factor += 2
            if num > 1:
                factors.add(num)
            return factors
        
        for i, num in enumerate(nums):
            factors = get_prime_factors(num)
            for f in factors:
                if f in factor_to_index:
                    dsu.union(i, factor_to_index[f])
                else:
                    factor_to_index[f] = i
        
        # After union operations, check if every index has the same parent as index 0.
        root = dsu.find(0)
        for i in range(1, n):
            if dsu.find(i) != root:
                return False
        
        return True