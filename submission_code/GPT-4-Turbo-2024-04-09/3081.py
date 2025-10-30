class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        stack = []
        for num in nums:
            if stack and stack[-1] < num:
                stack.pop()  # Remove the smaller element
            else:
                stack.append(num)
        return len(stack)