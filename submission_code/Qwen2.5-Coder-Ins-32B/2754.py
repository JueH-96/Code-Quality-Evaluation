from typing import List

class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        max_pos = float('-inf')
        min_neg = float('inf')
        neg_count = 0
        zero_count = 0
        result = 1
        has_positive = False
        
        for num in nums:
            if num > 0:
                result *= num
                max_pos = max(max_pos, num)
                has_positive = True
            elif num < 0:
                result *= num
                min_neg = min(min_neg, num)
                neg_count += 1
            else:
                zero_count += 1
        
        if neg_count % 2 != 0:
            if neg_count == 1 and not has_positive and zero_count > 0:
                return 0
            result //= min_neg
        
        if result == 1 and zero_count > 0 and not has_positive:
            return max(nums)
        
        return result