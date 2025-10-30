from typing import List

class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        # We need to check for any valid starting index a such that
        # both subarrays nums[a:a+k] and nums[a+k:a+2*k] are strictly increasing.
        # The starting index a should be such that a+2*k-1 < n, i.e., a <= n - 2*k.
        for a in range(n - 2 * k + 1):
            # Check first subarray: nums[a], ..., nums[a + k - 1]
            increasing_first = True
            for i in range(a, a + k - 1):
                if nums[i] >= nums[i + 1]:
                    increasing_first = False
                    break
            
            # If first subarray is not strictly increasing, move to next a.
            if not increasing_first:
                continue
            
            # Check second subarray: nums[a+k], ..., nums[a+2*k - 1]
            increasing_second = True
            for i in range(a + k, a + 2 * k - 1):
                if nums[i] >= nums[i + 1]:
                    increasing_second = False
                    break
            
            if increasing_first and increasing_second:
                return True
        
        return False