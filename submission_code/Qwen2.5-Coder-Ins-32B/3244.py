from typing import List
import math

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        min_val = min(nums)
        count_min_val = nums.count(min_val)
        
        for num in nums:
            if num % min_val != 0:
                return 1
        
        return (count_min_val + 1) // 2