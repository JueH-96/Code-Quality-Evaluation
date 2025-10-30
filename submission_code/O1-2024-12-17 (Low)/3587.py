class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # We'll use dynamic programming where dp[i][c] represents the maximum score
        # attainable after day i if the tourist ends day i in city c.
        
        # Initialize dp with -inf to signify unreachable states initially.
        dp = [[float('-inf')] * n for _ in range(k)]
        
        # Base case: on day 0, we can start in any city and choose to stay there.
        # dp[0][c] = stayScore[0][c]
        for c in range(n):
            dp[0][c] = stayScore[0][c]
        
        # Fill dp for the subsequent days
        for i in range(k - 1):
            for c in range(n):
                if dp[i][c] == float('-inf'):
                    continue
                # Option 1: Stay in the same city on day i+1
                dp[i+1][c] = max(dp[i+1][c], dp[i][c] + stayScore[i+1][c])
                
                # Option 2: Travel to a different city on day i+1
                for d in range(n):
                    if d != c:
                        dp[i+1][d] = max(dp[i+1][d], dp[i][c] + travelScore[c][d])
        
        # The answer is the maximum dp value on day k-1 across all cities
        return max(dp[k - 1])