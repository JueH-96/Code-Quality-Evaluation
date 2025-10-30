from math import isqrt
from typing import List

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        if n <= 1:
            return True
        
        # If any element is 1 (and there is more than one element), it cannot connect to others.
        if any(x == 1 for x in nums):
            return False
        
        max_val = max(nums)
        # Precompute smallest prime factor (spf) for every number up to max_val
        spf = list(range(max_val + 1))
        for i in range(2, isqrt(max_val) + 1):
            if spf[i] == i:  # i is prime
                for j in range(i * i, max_val + 1, i):
                    if spf[j] == j:
                        spf[j] = i

        # Union-Find setup
        parent = list(range(n))
        rank = [0] * n

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rootx = find(x)
            rooty = find(y)
            if rootx == rooty:
                return
            if rank[rootx] < rank[rooty]:
                parent[rootx] = rooty
            elif rank[rootx] > rank[rooty]:
                parent[rooty] = rootx
            else:
                parent[rooty] = rootx
                rank[rootx] += 1

        # Map each prime factor to an index where it was first encountered.
        factor_to_index = {}
        
        for i, num in enumerate(nums):
            # Factorize the current number using the spf sieve.
            temp = num
            factors = set()
            while temp > 1:
                factors.add(spf[temp])
                temp //= spf[temp]
            for factor in factors:
                if factor in factor_to_index:
                    union(i, factor_to_index[factor])
                else:
                    factor_to_index[factor] = i

        # Check if all indices are in the same connected component.
        root0 = find(0)
        for i in range(1, n):
            if find(i) != root0:
                return False
        
        return True