class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        max_coverage = 2000  # Since the maximum 2*i for i=1000 is 2000
        dp = [[float('inf')] * (max_coverage + 1) for _ in range(n + 2)]
        
        # Base case: when i exceeds n, cost is 0
        for c in range(max_coverage + 1):
            dp[n + 1][c] = 0
        
        # Fill the DP table from i = n down to 1
        for i in range(n, 0, -1):
            for c in range(max_coverage + 1):
                if i > c:
                    new_c = max(c, 2 * i)
                    if new_c > max_coverage:
                        new_c = max_coverage
                    dp[i][c] = prices[i - 1] + dp[i + 1][new_c]
                else:
                    option1 = dp[i + 1][c]
                    new_c_buy = max(c, 2 * i)
                    if new_c_buy > max_coverage:
                        new_c_buy = max_coverage
                    option2 = prices[i - 1] + dp[i + 1][new_c_buy]
                    dp[i][c] = min(option1, option2)
        
        return dp[1][0]