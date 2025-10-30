from collections import defaultdict

class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        dp = defaultdict(lambda: (0, 0))  # Each key stores (sum, count)
        total = 0
        for num in nums:
            v1 = num - 1
            v2 = num + 1
            # Calculate sum_prev and count_prev from v1 and v2
            sum_prev = (dp[v1][0] + dp[v2][0]) % MOD
            count_prev = (dp[v1][1] + dp[v2][1]) % MOD
            # Compute the contribution of the current num
            sum_contribution = (sum_prev + num * (count_prev + 1)) % MOD
            count_contribution = (count_prev + 1) % MOD
            # Add the contribution to the total sum
            total = (total + sum_contribution) % MOD
            # Update the dp for the current num
            dp[num] = (sum_contribution, count_contribution)
        return total % MOD