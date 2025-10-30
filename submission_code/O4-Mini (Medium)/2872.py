from typing import List

class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        # Start from the end of the array
        curr = nums[-1]
        ans = curr
        
        # Traverse from right to left, trying to merge when possible
        for x in reversed(nums[:-1]):
            if x <= curr:
                # If current element can be merged into the running sum
                curr += x
            else:
                # Otherwise reset the running sum to this element
                curr = x
            ans = max(ans, curr)
        
        return ans