from typing import List

class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        total = 0
        n = len(nums)
        for i in range(n):
            current = nums[i]
            left = i - k
            right = i + k
            is_good = True
            
            # Check left neighbor if exists
            if left >= 0:
                if current <= nums[left]:
                    is_good = False
            
            # Check right neighbor if exists
            if right < n:
                if current <= nums[right]:
                    is_good = False
            
            if is_good:
                total += current
        
        return total