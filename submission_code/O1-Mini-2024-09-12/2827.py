from typing import List
import math

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        if len(nums) == 1:
            return True
        if 1 in nums:
            return False
        
        # Sieve of Eratosthenes to get smallest prime factor
        MAX = max(nums)
        spf = [0] * (MAX + 1)
        for i in range(2, MAX + 1):
            if spf[i] == 0:
                spf[i] = i
                for j in range(i * i, MAX + 1, i):
                    if spf[j] == 0:
                        spf[j] = i
        
        # Union-Find implementation
        parent = list(range(len(nums)))
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        
        def union(u, v):
            pu, pv = find(u), find(v)
            if pu != pv:
                parent[pu] = pv
        
        prime_to_index = {}
        for idx, num in enumerate(nums):
            x = num
            factors = set()
            while x > 1:
                factors.add(spf[x])
                x //= spf[x]
            for prime in factors:
                if prime in prime_to_index:
                    union(idx, prime_to_index[prime])
                else:
                    prime_to_index[prime] = idx
        
        root = find(0)
        for i in range(1, len(nums)):
            if find(i) != root:
                return False
        return True