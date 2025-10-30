class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        stack = []
        for num in reversed(nums):
            current = num
            while stack and current <= stack[-1]:
                current += stack.pop()
            stack.append(current)
        return max(stack) if stack else 0