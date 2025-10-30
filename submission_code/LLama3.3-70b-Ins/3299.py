from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        max_length = 0
        num_set = set(nums)
        
        for num in num_set:
            k = 1
            current_num = num
            length = 1
            while current_num * current_num in num_set:
                current_num *= current_num
                k += 1
                length += 2
            max_length = max(max_length, length)
        
        return max_length