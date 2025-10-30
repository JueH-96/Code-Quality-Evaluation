from typing import List

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        min_val = min(nums)
        f = nums.count(min_val)
        return (f + 1) // 2