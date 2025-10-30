from typing import List

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        sorted_nums = sorted(nums)
        last = -float('inf')
        current_run = 0
        max_run = 0
        
        for num in sorted_nums:
            if num > last + 1:
                last = num
                current_run = 1
            elif num == last + 1:
                last = num
                current_run += 1
            elif num <= last:
                if num + 1 > last:
                    last = last + 1
                    current_run += 1
                else:
                    continue
            if current_run > max_run:
                max_run = current_run
                
        return max_run