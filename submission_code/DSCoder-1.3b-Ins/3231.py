class Solution:
    def minimumAddedCoins(self, coins: List[int], target: int) -> int:
        dp = [0] + [-1]*target
        for i in range(1, target+1):
            for coin in coins:
                if i >= coin:
                    dp[i] = max(dp[i], dp[i-coin] + 1)
        return max(0, dp[-1] - 1)