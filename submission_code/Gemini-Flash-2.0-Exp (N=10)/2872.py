class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        
        stack = []
        for num in nums:
            stack.append(num)
            while len(stack) >= 2 and stack[-2] <= stack[-1]:
                val = stack.pop()
                stack[-1] += val
        
        return max(stack)