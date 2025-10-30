from math import gcd
from collections import defaultdict

class Solution:
    def canTraverseAllPairs(self, nums: list[int]) -> bool:
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            rootX = find(x)
            rootY = find(y)
            if rootX != rootY:
                if rank[rootX] > rank[rootY]:
                    parent[rootY] = rootX
                elif rank[rootX] < rank[rootY]:
                    parent[rootX] = rootY
                else:
                    parent[rootY] = rootX
                    rank[rootX] += 1
        
        n = len(nums)
        parent = list(range(n))
        rank = [0] * n
        primes = defaultdict(list)
        
        # Sieve of Eratosthenes to find primes up to max(nums)
        max_num = max(nums)
        sieve = [True] * (max_num + 1)
        for p in range(2, int(max_num ** 0.5) + 1):
            if sieve[p]:
                for i in range(p * p, max_num + 1, p):
                    sieve[i] = False
        
        # Assign each number to its lowest prime factor
        for i, num in enumerate(nums):
            if sieve[num]:  # num is prime
                primes[num].append(i)
            else:
                for j in range(2, int(num ** 0.5) + 1):
                    if num % j == 0:
                        primes[j].append(i)
                        while num % j == 0:
                            num //= j
                        if sieve[num]:  # remaining num is prime
                            primes[num].append(i)
                            break
        
        # Union indices with common prime factors
        for indices in primes.values():
            for i in range(len(indices) - 1):
                union(indices[i], indices[i + 1])
        
        # Check if all numbers are connected
        root = find(0)
        for i in range(1, n):
            if find(i) != root:
                return False
        return True