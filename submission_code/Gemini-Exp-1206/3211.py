class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * (n + 1)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        stack = []
        for i in range(1, n + 1):
            j = i
            while stack and prefix_sum[i] - prefix_sum[stack[-1][0]] < stack[-1][1]:
                j, _ = stack.pop()
            dp[i] = dp[j - 1] + 1
            stack.append((j - 1, prefix_sum[i] - prefix_sum[j - 1]))
        return dp[n]