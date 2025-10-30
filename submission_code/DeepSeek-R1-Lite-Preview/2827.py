from typing import List
from collections import defaultdict

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        if len(nums) == 1:
            return True
        
        if 1 in nums:
            return False
        
        max_num = max(nums)
        SPF = [0] * (max_num + 1)
        for i in range(2, max_num + 1):
            if SPF[i] == 0:
                SPF[i] = i
                for j in range(i*i, max_num + 1, i):
                    if SPF[j] == 0:
                        SPF[j] = i
        
        def get_prime_factors(x):
            factors = set()
            while x > 1:
                factors.add(SPF[x])
                x //= SPF[x]
            return factors
        
        prime_to_indices = defaultdict(list)
        for idx, num in enumerate(nums):
            primes = get_prime_factors(num)
            for p in primes:
                prime_to_indices[p].append(idx)
        
        uf = UnionFind(len(nums))
        for indices in prime_to_indices.values():
            if len(indices) > 1:
                first = indices[0]
                for idx in indices[1:]:
                    uf.union(first, idx)
        
        root = uf.find(0)
        for i in range(1, len(nums)):
            if uf.find(i) != root:
                return False
        return True