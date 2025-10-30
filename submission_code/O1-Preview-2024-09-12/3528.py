class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        stack = []
        for i in range(n - 1, -1, -1):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            if not stack:
                dp[i] = (n - 1 - i) * nums[i]
            else:
                j = stack[-1]
                dp[i] = max((j - i) * nums[i] + dp[j], (n - 1 - i) * nums[i])
            stack.append(i)
        return dp[0]