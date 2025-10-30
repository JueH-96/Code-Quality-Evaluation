from typing import List

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        if n <= 1:
            return True
        
        max_val = max(nums)
        # Build smallest prime factor (SPF) table up to max_val
        spf = list(range(max_val + 1))
        for i in range(2, int(max_val**0.5) + 1):
            if spf[i] == i:  # i is prime
                for j in range(i * i, max_val + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        
        # Union-Find (Disjoint Set Union) with path compression and union by rank
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[ry] < rank[rx]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1
        
        # Map each prime to the first index where it appears
        prime_to_index = {}
        
        for i, num in enumerate(nums):
            x = num
            # factorize x using spf, gather unique primes
            primes = set()
            while x > 1:
                p = spf[x]
                primes.add(p)
                while x % p == 0:
                    x //= p
            # Union this index with any previous index sharing the same prime
            for p in primes:
                if p in prime_to_index:
                    union(i, prime_to_index[p])
                else:
                    prime_to_index[p] = i
        
        # Check if all indices are in the same component
        root0 = find(0)
        for i in range(1, n):
            if find(i) != root0:
                return False
        return True