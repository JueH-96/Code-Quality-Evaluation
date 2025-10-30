from typing import List

def sieve_for_smallest_prime_factor(limit):
    SPF = [0] * (limit + 1)
    for i in range(2, limit + 1):
        if SPF[i] == 0:
            SPF[i] = i
            for j in range(i * 2, limit + 1, i):
                if SPF[j] == 0:
                    SPF[j] = i
    return SPF

def factorize(n, SPF):
    factors = {}
    while n > 1:
        p = SPF[n]
        factors[p] = factors.get(p, 0) + 1
        n //= p
    return factors

def primes_with_odd_exponents(factors):
    return {p for p, exp in factors.items() if exp % 2 != 0}

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        # Precompute SPF up to a reasonable limit
        SPF_LIMIT = 10**7
        SPF = sieve_for_smallest_prime_factor(SPF_LIMIT)
        
        # Group sums by frozenset of primes with odd exponents
        group_sums = {}
        for num in nums:
            if num == 1:
                prime_set = frozenset()
            else:
                factors = factorize(num, SPF)
                prime_set = frozenset(primes_with_odd_exponents(factors))
            if prime_set not in group_sums:
                group_sums[prime_set] = 0
            group_sums[prime_set] += num
        
        # Find the maximum sum among all groups
        return max(group_sums.values())