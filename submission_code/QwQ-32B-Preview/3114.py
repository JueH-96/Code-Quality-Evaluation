from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_sum = 0
        for i in range(n):
            # Set the peak height
            heights = [0] * n
            heights[i] = maxHeights[i]
            
            # Set heights to the left of i
            for j in range(i-1, -1, -1):
                if maxHeights[j] >= heights[j+1]:
                    heights[j] = heights[j+1]
                else:
                    heights[j] = maxHeights[j]
            
            # Set heights to the right of i
            for j in range(i+1, n):
                if maxHeights[j] >= heights[j-1]:
                    heights[j] = heights[j-1]
                else:
                    heights[j] = maxHeights[j]
            
            # Calculate the sum of heights
            current_sum = sum(heights)
            if current_sum > max_sum:
                max_sum = current_sum
        return max_sum