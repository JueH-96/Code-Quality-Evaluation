class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # dp[day][city] will store the maximum points that can be achieved at the end of 'day' if the tourist is in 'city'
        dp = [[0] * n for _ in range(k)]
        
        # Initialize the first day with the stay scores in each city
        for city in range(n):
            dp[0][city] = stayScore[0][city]
        
        # Fill the dp table for subsequent days
        for day in range(1, k):
            for city in range(n):
                # Start by considering staying in the same city
                max_points = dp[day-1][city] + stayScore[day][city]
                
                # Consider moving from any other city to this city
                for prev_city in range(n):
                    if prev_city != city:
                        max_points = max(max_points, dp[day-1][prev_city] + travelScore[prev_city][city])
                
                # Update dp table with the maximum points for this city on this day
                dp[day][city] = max_points
        
        # The answer is the maximum points that can be achieved on the last day in any city
        return max(dp[k-1])