from typing import List

class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        max_len = 0
        n = len(nums)
        
        for l in range(n):
            if nums[l] % 2 != 0 or nums[l] > threshold:
                continue
                
            current_length = 1
            if current_length > max_len:
                max_len = current_length
                
            r = l + 1
            while r < n:
                if nums[r] > threshold:
                    break
                if (nums[r] % 2) == (nums[r-1] % 2):
                    break
                current_length += 1
                if current_length > max_len:
                    max_len = current_length
                r += 1
                
        return max_len