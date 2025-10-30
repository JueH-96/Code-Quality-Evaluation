from typing import List

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        # Precomputed prime‐factor sets for 1..10
        prime_factors = {
            1: set(),
            2: {2},
            3: {3},
            4: {2},
            5: {5},
            6: {2, 3},
            7: {7},
            8: {2},
            9: {3},
            10: {2, 5},
        }
        
        n = len(nums)
        # max length of a contiguous subarray in which no prime divisor repeats
        max_pairwise_coprime = 1
        
        for i in range(n):
            used_primes = set()
            length = 0
            for j in range(i, n):
                pf = prime_factors[nums[j]]
                # If this number shares a prime with the current window, we can't extend
                if any(p in used_primes for p in pf):
                    break
                # Otherwise include it
                used_primes |= pf
                length += 1
                if length > max_pairwise_coprime:
                    max_pairwise_coprime = length
        
        # Any subarray of length 2 is always product‐equivalent,
        # so the answer is at least 2 (n >= 2 by constraints).
        return max(max_pairwise_coprime, 2)