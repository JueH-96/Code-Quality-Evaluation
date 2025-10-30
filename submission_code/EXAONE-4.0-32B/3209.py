class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            r = min(n - 1, i + i + 1)
            start = i + 1
            end = r + 1
            if end > n:
                end = n
            min_val = float('inf')
            for j in range(start, end + 1):
                if dp[j] < min_val:
                    min_val = dp[j]
            dp[i] = prices[i] + min_val
        return dp[0]