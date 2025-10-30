from typing import List

class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        res = 0
        
        # Iterate over all possible split positions
        for i in range(1, n):
            # Check if the left subarray contains exactly one 1
            left_count = nums[:i].count(1)
            # Check if the right subarray contains exactly one 1
            right_count = nums[i:].count(1)
            
            # If both subarrays contain exactly one 1, increment the result
            if left_count == 1 and right_count == 1:
                res += 1
        
        # Return the result modulo 10^9 + 7
        return res % MOD