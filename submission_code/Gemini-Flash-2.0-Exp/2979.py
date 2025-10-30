class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        
        events = [[] for _ in range(n)]
        for start, end, gold in offers:
            events[end].append((start, gold))
        
        dp = [0] * n
        
        for i in range(n):
            if i > 0:
                dp[i] = dp[i-1]
            
            for start, gold in events[i]:
                if start == 0:
                    dp[i] = max(dp[i], gold)
                else:
                    dp[i] = max(dp[i], dp[start-1] + gold)
        
        if n > 0:
            return dp[n-1]
        else:
            return 0