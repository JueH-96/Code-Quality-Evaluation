class Solution:
    def sumOfPower(self, nums: list[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        total_power = 0

        for i in range(1 << n):
            subsequence = []
            for j in range(n):
                if (i >> j) & 1:
                    subsequence.append(nums[j])
            
            if not subsequence:
                continue

            dp = [[0] * (k + 1) for _ in range(len(subsequence) + 1)]
            dp[0][0] = 1

            for row in range(1, len(subsequence) + 1):
                for col in range(k + 1):
                    dp[row][col] = dp[row - 1][col]
                    if col >= subsequence[row - 1]:
                        dp[row][col] = (dp[row][col] + dp[row - 1][col - subsequence[row - 1]]) % MOD
            
            total_power = (total_power + dp[len(subsequence)][k]) % MOD

        return total_power