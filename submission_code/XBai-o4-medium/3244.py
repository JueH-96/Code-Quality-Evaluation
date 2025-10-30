import math
from typing import List
from functools import reduce

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        g = reduce(math.gcd, nums)
        count = 0
        all_equal = True
        for num in nums:
            if num != g:
                all_equal = False
            if num == g:
                count += 1
        if all_equal:
            return (len(nums) + 1) // 2
        else:
            if count == 0:
                return 1
            else:
                return (count + 1) // 2