class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        stack = []
        # Iterate from the end to the beginning
        for num in reversed(nums):
            # Merge with stack top if necessary
            while stack and num > stack[-1]:
                num += stack.pop()
            stack.append(num)
        # The stack now contains the segments in reverse order
        # The number of segments is the length of the stack
        return len(stack)