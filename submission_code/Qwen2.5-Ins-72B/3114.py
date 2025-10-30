from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        def calculate_sum(heights, peak):
            left = [heights[peak]]
            for i in range(peak - 1, -1, -1):
                left.append(min(left[-1], heights[i]))
            left.reverse()
            
            right = [heights[peak]]
            for i in range(peak + 1, len(heights)):
                right.append(min(right[-1], heights[i]))
            
            return sum(left) + sum(right) - heights[peak]
        
        max_sum = 0
        for i in range(len(maxHeights)):
            max_sum = max(max_sum, calculate_sum(maxHeights, i))
        
        return max_sum