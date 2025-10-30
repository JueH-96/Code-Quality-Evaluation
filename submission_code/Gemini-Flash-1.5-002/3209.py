class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        dp = {}
        def solve(i):
            if i >= n:
                return 0
            if i in dp:
                return dp[i]
            ans = float('inf')
            # Option 1: Buy the current fruit
            ans = min(ans, prices[i] + solve(i + i + 1))
            # Option 2: Try to get it for free from a previous purchase
            for j in range(max(0, i - i), i):
                ans = min(ans, solve(i + 1))

            dp[i] = ans
            return ans

        return solve(0)