from typing import List # Ensure List is imported for type hinting

class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.num_sets = n
        # For union by size optimization (optional but good practice)
        self.sz = [1] * n

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # Union by size: attach smaller tree's root under larger tree's root
            if self.sz[root_i] < self.sz[root_j]:
                root_i, root_j = root_j, root_i # Ensure root_i's component is larger or equal
            
            self.parent[root_j] = root_i
            self.sz[root_i] += self.sz[root_j]
            self.num_sets -= 1
            return True
        return False

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)
        if n == 1:
            return True

        # If any number is 1, it cannot connect to any other number (gcd(1, x) = 1).
        # So, if n > 1 and 1 is present, it's impossible for all pairs to be connected.
        for x in nums:
            if x == 1:
                return False 

        # Determine maximum value in nums to set Sieve range
        M = 0
        for x in nums:
            if x > M:
                M = x
        
        # Sieve for Smallest Prime Factor (SPF)
        # spf[k] will store the smallest prime factor of k.
        spf = list(range(M + 1)) 
        # spf[0] and spf[1] are not used for prime factorization logic.
        # Iterate i from 2 up to M:
        # If spf[i] == i, then i is prime. Mark its multiples.
        for i in range(2, M + 1):
            if spf[i] == i:  # i is prime
                # Mark multiples of i. Start from i*i as smaller multiples
                # (e.g., 2*i, 3*i) would have already been marked by smaller primes (2, 3, ...).
                for j in range(i * i, M + 1, i):
                    if spf[j] == j: # if j hasn't been marked by a smaller prime yet
                        spf[j] = i
        
        # Function to get distinct prime factors of a number using SPF array
        # Memoization could be used here if nums has many duplicates, but not strictly necessary.
        def get_prime_factors(num_val: int, spf_arr: List[int]) -> set[int]:
            factors = set()
            # All nums[i] are > 1 at this point due to the earlier check for 1s.
            while num_val > 1:
                factors.add(spf_arr[num_val])
                num_val //= spf_arr[num_val]
            return factors

        dsu = DSU(n)
        # prime_to_component_root maps a prime number to the root of the DSU component
        # that currently "contains" this prime (i.e., an index whose number has this prime factor).
        prime_to_component_root = {} 

        for i in range(n):
            num = nums[i]
            
            # Get prime factors for nums[i]
            prime_factors_for_num = get_prime_factors(num, spf)
            
            # For each prime factor p of nums[i]:
            # If p has been seen before (is in prime_to_component_root),
            # union the component of current index 'i' with the component associated with p.
            # This ensures that index 'i' becomes part of the larger component linked by p.
            for p in prime_factors_for_num:
                if p in prime_to_component_root:
                    dsu.union(i, prime_to_component_root[p])
                # If p is not in prime_to_component_root, it means this is the first time
                # this prime factor p is being associated with a component.
                # It will be associated with i's component in the loop below.
            
            # After all unions involving i and its prime factors' existing components are done,
            # find the final root of the component that i now belongs to.
            final_root_of_i = dsu.find(i)
            
            # All prime factors of nums[i] now "belong" to this same component (rooted at final_root_of_i).
            # Update the map so these primes point to this unified component's root.
            # This ensures that future numbers sharing these primes will connect to this common component.
            for p in prime_factors_for_num:
                prime_to_component_root[p] = final_root_of_i
        
        # If all indices are in the same DSU component, num_sets will be 1.
        return dsu.num_sets == 1