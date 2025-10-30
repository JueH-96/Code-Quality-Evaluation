from typing import List

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        min_val = min(nums)
        count = 0
        for num in nums:
            if num % min_val != 0:
                return 1
            if num == min_val:
                count += 1
        
        if count == 1:
            return 1
        
        return (count + 1) // 2