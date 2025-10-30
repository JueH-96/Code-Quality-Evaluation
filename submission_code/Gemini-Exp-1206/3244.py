from math import gcd
from typing import List

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        min_val = min(nums)
        count_min = nums.count(min_val)
        
        for num in nums:
            if num % min_val != 0:
                return 1
        
        return (count_min + 1) // 2