class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        ans = 0
        for i in range(n):
            left = [0] * n
            right = [0] * n
            stack = []
            for j in range(n):
                while stack and maxHeights[stack[-1]] > maxHeights[j]:
                    stack.pop()
                if not stack:
                    left[j] = maxHeights[j] * (j + 1)
                else:
                    left[j] = left[stack[-1]] + maxHeights[j] * (j - stack[-1])
                stack.append(j)
            stack = []
            for j in range(n - 1, -1, -1):
                while stack and maxHeights[stack[-1]] > maxHeights[j]:
                    stack.pop()
                if not stack:
                    right[j] = maxHeights[j] * (n - j)
                else:
                    right[j] = right[stack[-1]] + maxHeights[j] * (stack[-1] - j)
                stack.append(j)
            ans = max(ans, left[i] + right[i] - maxHeights[i])
        return ans