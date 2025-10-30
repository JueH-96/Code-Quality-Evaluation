from typing import List
from math import isqrt

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        """
        Each operation on a number x (x > 1 and composite) replaces x with its
        smallest prime factor (because we divide by the greatest proper divisor,
        which is x / smallest_prime_factor(x)).
        For primes (and for 1) an operation has no effect, so every element can
        effectively be changed at most once: either we leave it as-is or, if it
        is composite, we replace it by its smallest prime factor.

        Greedy strategy from right to left:
          • keep the current element if it is already ≤ the element to its right;
          • otherwise, try to perform the (only useful) operation once and check
            whether the obtained value fits.
        If neither option works, the task is impossible.

        We pre-compute the smallest prime factor (spf) for every integer up to
        max(nums) with a linear-time sieve.
        """
        if not nums:
            return 0

        max_val = max(nums)

        # Build array of smallest prime factor for every number up to max_val.
        spf = list(range(max_val + 1))
        if max_val >= 2:
            for p in range(2, isqrt(max_val) + 1):
                if spf[p] == p:                       # p is prime
                    for multiple in range(p * p, max_val + 1, p):
                        if spf[multiple] == multiple: # not assigned yet
                            spf[multiple] = p
        if max_val >= 1:
            spf[1] = 1
        spf[0] = 0

        next_val = float('inf')   # value that current element must not exceed
        operations = 0

        for x in reversed(nums):
            if x <= next_val:               # no need to change this element
                next_val = x
                continue

            # Try to reduce x once (only possible/or useful if x is composite)
            smallest_prime = spf[x]
            if smallest_prime == x:         # x is 1 or prime → cannot reduce
                return -1
            if smallest_prime <= next_val:  # one operation makes it fit
                operations += 1
                next_val = smallest_prime
            else:                           # even after reduction it is too big
                return -1

        return operations