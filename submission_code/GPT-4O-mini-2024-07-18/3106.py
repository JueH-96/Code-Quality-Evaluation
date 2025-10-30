class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = [[-1] * (target + 1) for _ in range(n + 1)]
        dp[0][0] = 0  # Base case: sum of 0 with 0 elements

        for i in range(1, n + 1):
            for j in range(target + 1):
                # If we don't take the current element
                dp[i][j] = dp[i - 1][j]
                # If we take the current element
                if j >= nums[i - 1] and dp[i - 1][j - nums[i - 1]] != -1:
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - nums[i - 1]] + 1)

        return dp[n][target] if dp[n][target] != -1 else -1