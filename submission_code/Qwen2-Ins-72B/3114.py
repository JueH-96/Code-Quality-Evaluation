from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        def calculate_sum(heights, stack):
            total_sum = 0
            for i in range(len(heights)):
                if stack and stack[-1][0] > i:
                    total_sum += min(stack[-1][1], heights[i])
                else:
                    total_sum += heights[i]
                    stack.append((i, heights[i]))
            return total_sum

        max_sum = 0
        n = len(maxHeights)
        for i in range(n):
            stack = []
            left_sum = calculate_sum(maxHeights[:i+1][::-1], stack)
            stack = []
            right_sum = calculate_sum(maxHeights[i:], stack)
            max_sum = max(max_sum, left_sum + right_sum - maxHeights[i])
        return max_sum