from typing import List

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        min_val = min(nums)
        count = nums.count(min_val)
        return max(1, count - nums.count(min_val * 2))