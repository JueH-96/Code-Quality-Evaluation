from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        left = [0] * n
        right = [0] * n
        stack = []
        
        # Calculate left array
        for i in range(n):
            while stack and maxHeights[stack[-1]] > maxHeights[i]:
                stack.pop()
            if stack:
                left[i] = left[stack[-1]] + (i - stack[-1]) * maxHeights[i]
            else:
                left[i] = (i + 1) * maxHeights[i]
            stack.append(i)
        
        stack = []
        
        # Calculate right array
        for i in range(n - 1, -1, -1):
            while stack and maxHeights[stack[-1]] > maxHeights[i]:
                stack.pop()
            if stack:
                right[i] = right[stack[-1]] + (stack[-1] - i) * maxHeights[i]
            else:
                right[i] = (n - i) * maxHeights[i]
            stack.append(i)
        
        # Calculate the maximum sum
        max_sum = 0
        for i in range(n):
            max_sum = max(max_sum, left[i] + right[i] - maxHeights[i])
        
        return max_sum