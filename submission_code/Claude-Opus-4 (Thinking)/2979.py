class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Group offers by their end position
        offers_by_end = [[] for _ in range(n)]
        for start, end, gold in offers:
            offers_by_end[end].append((start, gold))
        
        # dp[i] = maximum gold from houses 0 to i-1
        dp = [0] * (n + 1)
        
        for i in range(1, n + 1):
            # Option 1: Don't include any offer ending at house i-1
            dp[i] = dp[i-1]
            
            # Option 2: Include an offer ending at house i-1
            for start, gold in offers_by_end[i-1]:
                dp[i] = max(dp[i], dp[start] + gold)
        
        return dp[n]