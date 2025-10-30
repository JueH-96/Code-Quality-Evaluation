from typing import List

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7

        # Convert requirements into a more usable format
        req_dict = {}
        for end, cnt in requirements:
            req_dict[end] = cnt

        # Initialize a 2D DP array
        dp = [[0] * 401 for _ in range(n + 1)]
        dp[0][0] = 1

        # Initialize a prefix sum array for quick range sum queries
        prefix_sum = [[0] * 401 for _ in range(n + 1)]
        prefix_sum[0][0] = 1

        # Build the DP table
        for i in range(1, n + 1):
            for j in range(401):
                dp[i][j] = prefix_sum[i - 1][j]
                if j > 0:
                    dp[i][j] += prefix_sum[i - 1][j - 1]
                dp[i][j] %= MOD

            # Update the prefix sum array
            for j in range(401):
                prefix_sum[i][j] = (prefix_sum[i - 1][j] + dp[i][j]) % MOD

        # Calculate the result based on the requirements
        result = 0
        for end, cnt in requirements:
            result += dp[end + 1][cnt]
            result %= MOD

        return result