from typing import List

class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        sorted_nums = sorted(nums)
        left = 0
        max_beauty = 0
        for right in range(len(sorted_nums)):
            while sorted_nums[right] - sorted_nums[left] > 2 * k:
                left += 1
            max_beauty = max(max_beauty, right - left + 1)
        return max_beauty