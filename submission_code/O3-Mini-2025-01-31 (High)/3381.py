from typing import List

class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        # If k is 0, every non-empty subarray is special.
        if k == 0:
            return 1
        
        n = len(nums)
        res = float('inf')
        
        # Iterate over all possible starting indices of the subarray.
        for i in range(n):
            current_or = 0
            # Expand the subarray starting at index i.
            for j in range(i, n):
                current_or |= nums[j]
                # Check if the current subarray is special.
                if current_or >= k:
                    res = min(res, j - i + 1)
                    break  # No need to extend further from i.
        
        return res if res != float('inf') else -1