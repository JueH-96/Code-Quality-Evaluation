from typing import List

class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total = 0
        for i in range(n):
            left_index = i - k
            right_index = i + k
            
            is_good = True
            
            if left_index >= 0:
                if nums[i] <= nums[left_index]:
                    is_good = False
            
            if right_index < n:
                if nums[i] <= nums[right_index]:
                    is_good = False
            
            if is_good:
                total += nums[i]
                
        return total