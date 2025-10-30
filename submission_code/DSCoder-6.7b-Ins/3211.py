class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        stack = [-1]
        res = 0
        for i in range(n):
            if i > 0 and nums[i] < nums[i-1]:
                stack = [-1]
            while stack and nums[stack[-1]] < nums[i]:
                j = stack.pop()
                res = max(res, i - stack[-1])
            stack.append(i)
        return res