from typing import List

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        if n == 1:
            return True
        if 1 in nums:
            return False
        
        max_num = max(nums)
        # Build smallest prime factors (SPF) sieve
        spf = list(range(max_num + 1))
        for i in range(2, int(max_num**0.5) + 1):
            if spf[i] == i:  # i is a prime
                for j in range(i * i, max_num + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        
        def get_prime_factors(x):
            factors = set()
            while x != 1:
                p = spf[x]
                factors.add(p)
                while x % p == 0:
                    x = x // p
            return factors
        
        primes_set = set()
        for num in nums:
            factors = get_prime_factors(num)
            primes_set.update(factors)
        
        parent = {}
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]  # Path compression
                u = parent[u]
            return u
        
        def union(u, v):
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                parent[root_v] = root_u
        
        for num in nums:
            factors = get_prime_factors(num)
            primes_list = list(factors)
            if not primes_list:
                continue
            for p in primes_list:
                if p not in parent:
                    parent[p] = p
            first_p = primes_list[0]
            for p in primes_list[1:]:
                union(first_p, p)
        
        if not primes_set:
            return False
        
        root = find(next(iter(primes_set)))
        for p in primes_set:
            if find(p) != root:
                return False
        return True