from typing import List

class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        max_length = 1
        inc = [1] * n
        dec = [1] * n
        
        for i in range(1, n):
            if nums[i] > nums[i-1]:
                inc[i] = inc[i-1] + 1
                dec[i] = 1
            elif nums[i] < nums[i-1]:
                dec[i] = dec[i-1] + 1
                inc[i] = 1
            else:
                inc[i] = 1
                dec[i] = 1
            
            current_max = max(inc[i], dec[i])
            if current_max > max_length:
                max_length = current_max
        
        return max_length