from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_num = max(nums)
        count = 0
        freq = 0
        result = 0
        left = 0
        
        for right in range(len(nums)):
            if nums[right] == max_num:
                freq += 1
            
            while freq == k:
                if nums[left] == max_num:
                    freq -= 1
                left += 1
            
            result += left
        
        return result