from typing import List

class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [0] * (n + 1)

        for i in range(1, n + 1):
            dp[i] = dp[i - 1] + prices[i - 1]
            if i - 1 >= 0:
                dp[i] = min(dp[i], dp[i - 1])
            if i > 1:
                dp[i] = min(dp[i], dp[i - 2] + prices[i - 1])

        return dp[n]

# Example usage:
# solution = Solution()
# print(solution.minimumCoins([3, 1, 2]))  # Output: 4
# print(solution.minimumCoins([1, 10, 1, 1]))  # Output: 2