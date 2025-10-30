from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_sum = 0
        for i in range(n):
            heights = [0] * n
            heights[i] = maxHeights[i]
            # Left pass from i-1 down to 0
            for j in range(i - 1, -1, -1):
                heights[j] = min(maxHeights[j], heights[j + 1])
            # Right pass from i+1 to n-1
            for j in range(i + 1, n):
                heights[j] = min(maxHeights[j], heights[j - 1])
            current_sum = sum(heights)
            if current_sum > max_sum:
                max_sum = current_sum
        return max_sum