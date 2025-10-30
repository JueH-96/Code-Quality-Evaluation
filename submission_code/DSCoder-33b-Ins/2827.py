from typing import List
from math import gcd
from collections import defaultdict

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        def find(x):
            if x != parent[x]:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            parent[find(x)] = find(y)

        n = len(nums)
        parent = list(range(n))
        primes = defaultdict(list)

        for i, num in enumerate(nums):
            for p in range(2, int(num**0.5) + 1):
                if num % p == 0:
                    while num % p == 0:
                        num //= p
                    primes[p].append(i)
            if num > 1:
                primes[num].append(i)

        for group in primes.values():
            for i in range(1, len(group)):
                union(group[i - 1], group[i])

        return all(find(i) == find(0) for i in range(n))