class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # dp[day][city] = max points if we are in city 'city' at the end of day 'day'
        dp = [[-float('inf')] * n for _ in range(k)]
        
        # Day 0: we can start in any city and either stay or move
        for j in range(n):
            # Option 1: Start in city j and stay
            dp[0][j] = stayScore[0][j]
            # Option 2: Start in any city i and move to j
            for i in range(n):
                dp[0][j] = max(dp[0][j], travelScore[i][j])
        
        # Fill the dp table for subsequent days
        for day in range(1, k):
            for j in range(n):
                # Option 1: Was in city j yesterday and stay
                dp[day][j] = dp[day-1][j] + stayScore[day][j]
                # Option 2: Was in any city i yesterday and move to j
                for i in range(n):
                    dp[day][j] = max(dp[day][j], dp[day-1][i] + travelScore[i][j])
        
        # The answer is the maximum points we can get on the last day in any city
        return max(dp[k-1])