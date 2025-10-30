class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        stack = []
        for num in nums:
            while stack and stack[-1] <= num:
                num += stack.pop()
            stack.append(num)
        return max(stack)