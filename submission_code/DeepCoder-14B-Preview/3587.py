class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # Initialize the DP table with negative infinity
        dp = [[-float('inf')] * n for _ in range(k)]
        
        # Initialize the first day (i=0)
        for j in range(n):
            max_move = max(travelScore[s][j] for s in range(n))
            dp[0][j] = max(stayScore[0][j], max_move)
        
        # Fill the rest of the days
        for i in range(1, k):
            for j in range(n):
                # Option 1: Stay in the current city j
                option1 = dp[i-1][j] + stayScore[i][j] if dp[i-1][j] != -float('inf') else -float('inf')
                
                # Option 2: Move from any city m to current city j
                max_option2 = -float('inf')
                for m in range(n):
                    if dp[i-1][m] != -float('inf'):
                        current = dp[i-1][m] + travelScore[m][j]
                        if current > max_option2:
                            max_option2 = current
                
                # Determine the maximum points for the current day and city
                dp[i][j] = max(option1, max_option2) if max(option1, max_option2) != -float('inf') else -float('inf')
        
        # The answer is the maximum value in the last row of the DP table
        return max(dp[k-1]) if max(dp[k-1]) != -float('inf') else 0