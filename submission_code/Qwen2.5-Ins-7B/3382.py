from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        max_val = max(nums)
        count = 0
        subarray_count = 0
        
        for num in nums:
            if num == max_val:
                count += 1
                subarray_count += count
            else:
                count = 0
        
        return subarray_count