class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        if n <= 2:
            return True

        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]

        dp = [[False] * n for _ in range(n)]
        for length in range(1, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if length == 1:
                    dp[i][j] = True
                else:
                    for k in range(i, j):
                        left_sum = prefix_sum[k + 1] - prefix_sum[i]
                        right_sum = prefix_sum[j + 1] - prefix_sum[k + 1]
                        if (left_sum >= m or dp[i][k]) and (right_sum >= m or dp[k + 1][j]):
                            dp[i][j] = True
                            break

        return dp[0][n - 1]