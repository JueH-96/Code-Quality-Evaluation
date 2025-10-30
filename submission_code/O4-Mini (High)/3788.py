from typing import List

class Solution:
    def maxSum(self, nums: List[int]) -> int:
        # We can pick at most one occurrence of each value (subarray must have unique elements),
        # and we can skip any elements we like (via deletions). So to maximize sum we take
        # all distinct positive values.  If there are no positives, we must pick at least one
        # element (cannot end up empty), so we return the maximum element (which may be 0 or negative).
        uniq = set(nums)
        pos_sum = sum(x for x in uniq if x > 0)
        return pos_sum if pos_sum > 0 else max(nums)