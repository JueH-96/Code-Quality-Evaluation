from typing import List

class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        prev_sum = 0
        current_sum = 0
        count = 0
        for num in nums:
            current_sum += num
            if current_sum >= prev_sum:
                count += 1
                prev_sum = current_sum
                current_sum = 0
        return count