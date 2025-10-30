from typing import List

class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        # Precompute prime status for numbers up to 100
        max_val = 100
        is_prime = [True] * (max_val + 1)
        is_prime[0] = is_prime[1] = False
        for p in range(2, int(max_val**0.5) + 1):
            if is_prime[p]:
                for multiple in range(p*p, max_val+1, p):
                    is_prime[multiple] = False
        
        first_idx = None
        last_idx = None
        
        for i, num in enumerate(nums):
            if is_prime[num]:
                if first_idx is None:
                    first_idx = i
                last_idx = i
        
        # Since there's at least one prime, first_idx and last_idx are set
        return last_idx - first_idx