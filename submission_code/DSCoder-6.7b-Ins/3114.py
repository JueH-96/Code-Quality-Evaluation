class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        left, right = [0]*n, [n-1]*n
        stack = []

        # Find the next greater element for each element in the array
        for i in range(n):
            while stack and maxHeights[stack[-1]] < maxHeights[i]:
                right[stack.pop()] = i
            stack.append(i)

        # Empty the stack
        while stack:
            right[stack.pop()] = n

        # Find the previous greater element for each element in the array
        for i in range(n-1, -1, -1):
            while stack and maxHeights[stack[-1]] < maxHeights[i]:
                left[stack.pop()] = i
            stack.append(i)

        # Empty the stack
        while stack:
            left[stack.pop()] = -1

        # Calculate the maximum sum of heights
        max_sum = 0
        for i in range(n):
            max_sum = max(max_sum, maxHeights[i]*(right[i]-left[i]-1))

        return max_sum