from typing import List

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        # Count how many times the minimum element appears
        mn = min(nums)
        cnt = nums.count(mn)
        # The minimum final length is ceil(cnt/2)
        return (cnt + 1) // 2