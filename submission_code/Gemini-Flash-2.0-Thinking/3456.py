class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [[0] * (k + 1) for _ in range(n)]
        max_len = 0

        for i in range(n):
            dp[i][0] = 1  # Subsequence of length 1 ending at i has 0 differences
            for j in range(k + 1):
                dp[i][j] = 1  # Initialize with length 1

            for p in range(i):
                for j in range(k + 1):
                    # Extend subsequence ending at p
                    if nums[i] == nums[p]:
                        dp[i][j] = max(dp[i][j], dp[p][j] + 1)
                    elif j > 0:
                        dp[i][j] = max(dp[i][j], dp[p][j - 1] + 1)

            max_len = max(max_len, max(dp[i]))

        return max_len