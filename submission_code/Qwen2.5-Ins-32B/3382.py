from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        count = 0
        max_count = {}
        current_max = 0
        
        for num in nums:
            if num > current_max:
                current_max = num
                max_count = {num: 1}
                count += 1
            elif num == current_max:
                max_count[num] += 1
                count += max_count[num]
            else:
                count += 1
        
        return count