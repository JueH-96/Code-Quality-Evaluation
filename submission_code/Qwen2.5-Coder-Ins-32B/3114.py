from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        
        def calculate_left_sums():
            left_sums = [0] * n
            stack = []
            for i in range(n):
                while stack and maxHeights[stack[-1]] >= maxHeights[i]:
                    stack.pop()
                if stack:
                    left_sums[i] = left_sums[stack[-1]] + (i - stack[-1]) * maxHeights[i]
                else:
                    left_sums[i] = (i + 1) * maxHeights[i]
                stack.append(i)
            return left_sums
        
        def calculate_right_sums():
            right_sums = [0] * n
            stack = []
            for i in range(n - 1, -1, -1):
                while stack and maxHeights[stack[-1]] >= maxHeights[i]:
                    stack.pop()
                if stack:
                    right_sums[i] = right_sums[stack[-1]] + (stack[-1] - i) * maxHeights[i]
                else:
                    right_sums[i] = (n - i) * maxHeights[i]
                stack.append(i)
            return right_sums
        
        left_sums = calculate_left_sums()
        right_sums = calculate_right_sums()
        
        max_sum = 0
        for i in range(n):
            max_sum = max(max_sum, left_sums[i] + right_sums[i] - maxHeights[i])
        
        return max_sum