from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        max_num = max(nums)
        count = 0
        i = 0
        
        while i < n:
            if nums[i] == max_num:
                j = i
                while j < n and nums[j] == max_num:
                    j += 1
                length = j - i
                count += (length * (length + 1)) // 2
                i = j
            else:
                i += 1
        
        return count