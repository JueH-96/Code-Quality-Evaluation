from typing import List

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        def digit_difference(a: int, b: int) -> int:
            return sum(1 for x, y in zip(str(a), str(b)) if x != y)
        
        n = len(nums)
        total_diff = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                total_diff += digit_difference(nums[i], nums[j])
        
        return total_diff