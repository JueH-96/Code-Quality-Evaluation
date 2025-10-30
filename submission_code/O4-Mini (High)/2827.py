from typing import List

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        # Union-Find (Disjoint Set Union) implementation
        n = len(nums)
        if n <= 1:
            return True

        # Build smallest prime factor (SPF) sieve up to max(nums)
        max_val = max(nums)
        spf = list(range(max_val + 1))  # spf[x] will be the smallest prime factor of x
        for i in range(2, int(max_val**0.5) + 1):
            if spf[i] == i:  # i is prime
                for j in range(i * i, max_val + 1, i):
                    if spf[j] == j:
                        spf[j] = i

        # Union-Find data structures
        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int):
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

        # Map each prime factor to the first index where it appears
        prime_to_index = {}

        # Process each number: factor it, then union indices sharing primes
        for i, val in enumerate(nums):
            x = val
            # Extract unique prime factors of x
            factors = set()
            while x > 1:
                p = spf[x]
                factors.add(p)
                while x % p == 0:
                    x //= p
            # Union with any previous index that had the same prime factor
            for p in factors:
                if p in prime_to_index:
                    union(i, prime_to_index[p])
                else:
                    prime_to_index[p] = i

        # Check if all indices are in the same connected component
        root0 = find(0)
        for i in range(1, n):
            if find(i) != root0:
                return False
        return True