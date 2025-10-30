class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        left = [0] * n
        right = [0] * n
        stack = []
        
        # Calculate left boundary
        for i in range(n):
            while stack and maxHeights[stack[-1]] >= maxHeights[i]:
                stack.pop()
            left[i] = i - (stack[-1] + 1) if stack else i + 1
            stack.append(i)
        
        stack = []
        # Calculate right boundary
        for i in range(n - 1, -1, -1):
            while stack and maxHeights[stack[-1]] >= maxHeights[i]:
                stack.pop()
            right[i] = (stack[-1] - i - 1) if stack else n - i - 1
            stack.append(i)
        
        # Calculate the maximum sum of heights
        return max(maxHeights[i] * (left[i] + right[i]) for i in range(n))