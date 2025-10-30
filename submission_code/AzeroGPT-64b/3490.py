class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        max_length = 0
        dp = [0, 0, 0, 0]

        for i in range(n):
            a = (dp[0] + dp[1] + 2 + 1) % 2
            b = (dp[1] + dp[2] + 2 + 1) % 2
            if (nums[i] + dp[0]) % 2 == a:
                dp[1] = dp[0] + 1
            if (nums[i] + dp[1]) % 2 == b:
                dp[2] = max(dp[2], dp[1] + 1)
            if (nums[i] + dp[2]) % 2 == a:
                dp[3] = max(dp[3], dp[2] + 1)
            max_length = max(max_length, dp[1], dp[2], dp[3])
            dp[0] = dp[1]

        return max_length