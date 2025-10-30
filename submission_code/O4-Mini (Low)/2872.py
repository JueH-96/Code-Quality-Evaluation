from typing import List

class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        # We traverse from right to left, merging whenever nums[i] <= the current merged suffix value.
        # next_val holds the sum of the "current suffix" that we can merge into.
        next_val = nums[-1]
        ans = next_val
        
        # Iterate from second-last element down to the first
        for i in range(len(nums) - 2, -1, -1):
            if nums[i] <= next_val:
                # We can merge nums[i] into the suffix
                next_val += nums[i]
            else:
                # Can't merge, start a new suffix
                next_val = nums[i]
            # Keep track of the maximum element obtainable
            ans = max(ans, next_val)
        
        return ans