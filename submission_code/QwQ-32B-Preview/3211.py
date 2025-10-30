from typing import List

class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        segments = []
        current_sum = 0
        for num in nums:
            current_sum += num
            if len(segments) == 0 or current_sum >= segments[-1]:
                segments.append(current_sum)
                current_sum = 0
        # If there is any remaining sum, add it to the segments
        if current_sum > 0:
            segments.append(current_sum)
        
        return len(segments)