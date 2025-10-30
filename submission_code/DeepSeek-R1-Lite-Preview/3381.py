from typing import List

class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        if k == 0:
            return 1  # Any non-empty subarray satisfies the condition
        
        n = len(nums)
        min_length = float('inf')  # Initialize with infinity to find the minimum
        
        for i in range(n):
            current_or = 0
            for j in range(i, n):
                current_or |= nums[j]
                if current_or >= k:
                    min_length = min(min_length, j - i + 1)
                    break  # No need to consider larger subarrays starting at i
        
        return min_length if min_length != float('inf') else -1