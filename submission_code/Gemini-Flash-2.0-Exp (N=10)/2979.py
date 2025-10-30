class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        
        offers.sort(key=lambda x: x[1])
        
        dp = [0] * n
        
        for i in range(n):
            
            if i > 0:
                dp[i] = dp[i-1]
            
            for start, end, gold in offers:
                if end == i:
                    
                    prev_profit = 0
                    if start > 0:
                        prev_profit = dp[start-1]
                    
                    dp[i] = max(dp[i], prev_profit + gold)
        
        return dp[n-1] if n > 0 else 0