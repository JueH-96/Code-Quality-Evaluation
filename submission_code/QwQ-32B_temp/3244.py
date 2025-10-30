from typing import List

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        m = min(nums)
        count = 0
        has_non_divisible = False
        for x in nums:
            if x % m != 0:
                has_non_divisible = True
            if x == m:
                count += 1
        if has_non_divisible:
            return 1
        else:
            return (count + 1) // 2