from typing import List
import itertools

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        total_power = 0
        
        # Generate all possible subsequences of length k
        for subsequence in itertools.combinations(nums, k):
            # Calculate the power of the current subsequence
            subsequence_power = min(abs(a - b) for i, a in enumerate(subsequence) for b in subsequence[i+1:])
            
            # Add the power of the current subsequence to the total power
            total_power += subsequence_power
        
        # Return the total power modulo MOD
        return total_power % MOD