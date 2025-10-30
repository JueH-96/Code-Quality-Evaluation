from typing import List

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        # With only one element there are no pairs to connect
        if n == 1:
            return True
        
        # If any number is 1 and there is more than one element,
        # that index can never connect with the others.
        if 1 in nums:
            return False
        
        # --------  Union-Find (Disjoint Set) structure  --------
        class DSU:
            def __init__(self, size: int):
                self.parent = list(range(size))
                self.sz     = [1]*size
            
            def find(self, x: int) -> int:
                while self.parent[x] != x:
                    self.parent[x] = self.parent[self.parent[x]]
                    x = self.parent[x]
                return x
            
            def union(self, x: int, y: int):
                rx, ry = self.find(x), self.find(y)
                if rx == ry:
                    return
                if self.sz[rx] < self.sz[ry]:
                    rx, ry = ry, rx
                self.parent[ry] = rx
                self.sz[rx] += self.sz[ry]
        
        dsu = DSU(n)
        
        # --------  Pre–compute smallest prime factor (SPF)  --------
        max_val = max(nums)
        spf = list(range(max_val + 1))          # spf[x] = smallest prime factor of x
        for i in range(2, int(max_val**0.5) + 1):
            if spf[i] == i:                     # i is prime
                for j in range(i*i, max_val+1, i):
                    if spf[j] == j:
                        spf[j] = i
        
        def prime_factors(x: int) -> List[int]:
            """Return the list of distinct prime factors of x using the SPF table."""
            factors = []
            while x > 1:
                p = spf[x]
                factors.append(p)
                while x % p == 0:
                    x //= p
            return factors
        
        # For each prime factor remember the first index where it appeared
        first_idx_for_prime = dict()   # prime -> index
        
        for idx, val in enumerate(nums):
            for p in set(prime_factors(val)):   # distinct primes only
                if p in first_idx_for_prime:
                    dsu.union(idx, first_idx_for_prime[p])
                else:
                    first_idx_for_prime[p] = idx
        
        # Check if all indices belong to the same connected component
        root0 = dsu.find(0)
        for i in range(1, n):
            if dsu.find(i) != root0:
                return False
        return True