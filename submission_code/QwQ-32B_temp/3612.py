from typing import List

class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        is_increasing = [False] * n  # Initialize array to track if subarray starts at s is increasing
        
        # Precompute whether each subarray of length k is strictly increasing
        for s in range(n - k + 1):
            valid = True
            for i in range(s, s + k - 1):
                if nums[i] >= nums[i + 1]:
                    valid = False
                    break
            is_increasing[s] = valid
        
        # Check for any valid a where both a and a+k subarrays are increasing
        for a in range(n - 2 * k + 1):
            if is_increasing[a] and is_increasing[a + k]:
                return True
        
        return False