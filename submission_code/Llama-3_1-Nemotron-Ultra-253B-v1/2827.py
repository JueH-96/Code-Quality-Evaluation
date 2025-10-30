from typing import List

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        # Check if there's a 1 in the array and the length is more than 1
        if len(nums) == 1:
            return True
        if any(num == 1 for num in nums):
            return False
        
        max_num = max(nums)
        if max_num < 2:
            return True  # All elements are 1, but checked earlier
        
        # Precompute smallest prime factors (SPF) up to max_num
        spf = list(range(max_num + 1))
        for i in range(2, int(max_num ** 0.5) + 1):
            if spf[i] == i:  # i is a prime
                for j in range(i * i, max_num + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        
        def get_primes(n):
            primes = set()
            while n != 1:
                p = spf[n]
                primes.add(p)
                while n % p == 0:
                    n //= p
            return primes
        
        parent = {}
        
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        
        def union(u, v):
            u_root = find(u)
            v_root = find(v)
            if u_root != v_root:
                parent[v_root] = u_root
        
        all_primes = set()
        for num in nums:
            primes = get_primes(num)
            if not primes:
                continue  # num is 1, but already checked
            primes_list = list(primes)
            for p in primes_list:
                if p not in parent:
                    parent[p] = p
            if len(primes_list) > 1:
                first = primes_list[0]
                for p in primes_list[1:]:
                    union(first, p)
            all_primes.update(primes_list)
        
        if not all_primes:
            return True  # All elements are 1, but checked earlier
        
        root = find(next(iter(all_primes)))
        for p in all_primes:
            if find(p) != root:
                return False
        return True