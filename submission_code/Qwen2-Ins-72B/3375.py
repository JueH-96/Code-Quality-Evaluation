class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        coins.sort()
        dp = [0] + [float('inf')] * k
        for coin in coins:
            for j in range(coin, k + 1):
                dp[j] = min(dp[j], dp[j - coin] + coin)
        return dp[k]