from typing import List

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        # Handle the case where there are 1s in the array
        if 1 in nums:
            n = len(nums)
            if n == 1:
                return True
            # Check if all elements are 1
            all_ones = all(x == 1 for x in nums)
            return not all_ones  # If not all ones, return False
        
        # Precompute smallest prime factors (SPF) up to the maximum number in nums
        max_num = max(nums)
        max_spf = max(max_num, 2)  # Ensure at least 2 to handle the SPF array
        spf = list(range(max_spf + 1))
        for i in range(2, int(max_spf ** 0.5) + 1):
            if spf[i] == i:  # i is a prime
                for j in range(i * i, max_spf + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        
        # Function to get the unique prime factors of a number
        def get_primes(x: int) -> set:
            primes = set()
            while x != 1:
                p = spf[x]
                primes.add(p)
                while x % p == 0:
                    x //= p
            return primes
        
        parent = {}
        
        def find(u: int) -> int:
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]
        
        def union(u: int, v: int) -> None:
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                parent[root_v] = root_u
        
        primes_set = set()
        for num in nums:
            primes = get_primes(num)
            if not primes:
                continue  # This should not happen as num > 1
            primes_list = list(primes)
            # Connect all primes in this number's factors
            first = primes_list[0]
            for p in primes_list[1:]:
                if p not in parent:
                    parent[p] = p
                union(first, p)
            primes_set.update(primes_list)
        
        # If there are no primes (handled by earlier checks, but included for safety)
        if not primes_set:
            return True
        
        # Check if all primes are in the same connected component
        root = find(next(iter(primes_set)))
        for p in primes_set:
            if find(p) != root:
                return False
        return True