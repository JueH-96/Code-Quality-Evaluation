from typing import List

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        
        nums.sort()
        product = 1
        negative_count = 0
        negative_product = 1
        zero_count = 0
        
        for num in nums:
            if num < 0:
                negative_count += 1
                negative_product *= num
            elif num > 0:
                product *= num
            else:
                zero_count += 1
        
        if negative_count % 2 == 0:
            product *= negative_product
        elif negative_count > 1:
            product *= negative_product // nums[0]
        
        if product == 1 and zero_count > 0 and negative_count < 2:
            return 0
        
        return product