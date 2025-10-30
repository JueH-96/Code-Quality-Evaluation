from typing import List

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        min_val = min(nums)
        for num in nums:
            if num % min_val != 0:
                return 1
        return (nums.count(min_val) + 1) // 2