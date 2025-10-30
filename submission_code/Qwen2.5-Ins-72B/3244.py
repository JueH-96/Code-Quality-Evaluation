from typing import List
import math

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        min_num = min(nums)
        gcd = nums[0]
        
        for num in nums:
            gcd = math.gcd(gcd, num)
        
        if gcd != min_num:
            return 1
        
        count_min = nums.count(min_num)
        return max(1, (count_min + 1) // 2)