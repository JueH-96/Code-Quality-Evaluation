from typing import List

class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        """
        Scan the array from right to left.
        Keep a running value `cur`, initially equal to the last element.
        For every preceding element:
            • If nums[i] <= cur, we are allowed to merge it with `cur`,
              so we add it to `cur`.
            • Otherwise, merging cannot be done, so a new segment starts
              and `cur` becomes nums[i].
        The maximum value reached by `cur` during this process is the
        largest element we can obtain in any final array.
        """
        cur = nums[-1]          # value of the current (right-most) segment
        best = cur              # best value seen so far
        
        for i in range(len(nums) - 2, -1, -1):
            if nums[i] <= cur:  # we can merge nums[i] into the segment
                cur += nums[i]
            else:               # cannot merge, start new segment
                cur = nums[i]
            best = max(best, cur)
        
        return best