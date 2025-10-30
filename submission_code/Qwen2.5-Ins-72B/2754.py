from typing import List
import math

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        
        positive = []
        negative = []
        
        for num in nums:
            if num > 0:
                positive.append(num)
            elif num < 0:
                negative.append(num)
        
        negative.sort()
        
        if len(negative) % 2 != 0:
            negative.pop()
        
        max_strength = 1
        if positive:
            for num in positive:
                max_strength *= num
        
        if negative:
            for num in negative:
                max_strength *= num
        
        return max_strength if max_strength != 1 else max(nums)