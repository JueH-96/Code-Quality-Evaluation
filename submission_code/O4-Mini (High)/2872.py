from typing import List

class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        # Start from the rightmost element
        cur = nums[-1]
        ans = cur
        
        # Traverse from second-last to the first
        for i in range(len(nums) - 2, -1, -1):
            if nums[i] <= cur:
                # We can merge nums[i] into the current suffix sum
                cur += nums[i]
            else:
                # Cannot merge: record the current sum as a candidate answer
                ans = max(ans, cur)
                # Reset cur to the standalone nums[i]
                cur = nums[i]
        
        # Final check to cover the last accumulated segment
        ans = max(ans, cur)
        return ans