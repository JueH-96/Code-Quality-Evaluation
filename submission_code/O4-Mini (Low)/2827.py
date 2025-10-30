from typing import List

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        if n <= 1:
            return True

        # Find max value for sieve limit
        max_val = max(nums)

        # Build smallest prime factor (spf) sieve up to max_val
        spf = list(range(max_val + 1))
        for i in range(2, int(max_val**0.5) + 1):
            if spf[i] == i:
                for j in range(i * i, max_val + 1, i):
                    if spf[j] == j:
                        spf[j] = i

        # Union-Find (Disjoint Set Union) implementation
        parent = list(range(n))
        size = [1] * n

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if size[rx] < size[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            size[rx] += size[ry]

        # Map each prime factor to the first index that had it
        prime_to_index = {}

        # For each number, factorize by spf and union indices sharing a prime
        for idx, val in enumerate(nums):
            x = val
            # Factorize val into unique primes
            last_p = 0
            while x > 1:
                p = spf[x]
                x //= p
                if p == last_p:
                    continue
                last_p = p
                if p in prime_to_index:
                    union(idx, prime_to_index[p])
                else:
                    prime_to_index[p] = idx

        # Finally check that all indices are connected to index 0
        root0 = find(0)
        for i in range(1, n):
            if find(i) != root0:
                return False
        return True