class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        left = [0] * n
        stack = []
        for i in range(n):
            while stack and maxHeights[stack[-1]] > maxHeights[i]:
                stack.pop()
            if stack:
                left[i] = left[stack[-1]] + maxHeights[i] * (i - stack[-1])
            else:
                left[i] = maxHeights[i] * (i + 1)
            stack.append(i)
        
        right = [0] * n
        stack = []
        for i in range(n - 1, -1, -1):
            while stack and maxHeights[stack[-1]] > maxHeights[i]:
                stack.pop()
            if stack:
                right[i] = right[stack[-1]] + maxHeights[i] * (stack[-1] - i)
            else:
                right[i] = maxHeights[i] * (n - i)
            stack.append(i)
        
        return max(left[i] + right[i] - maxHeights[i] for i in range(n))