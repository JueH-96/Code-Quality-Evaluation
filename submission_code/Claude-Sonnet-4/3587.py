class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # dp[day][city] = maximum score achievable by being in city on day
        dp = [[-1] * n for _ in range(k)]
        
        # Base case: Day 0 - can start in any city
        # On day 0, we must stay in the starting city
        for city in range(n):
            dp[0][city] = stayScore[0][city]
        
        # Fill dp table for remaining days
        for day in range(1, k):
            for curr_city in range(n):
                # Option 1: Stay in current city
                stay_score = dp[day-1][curr_city] + stayScore[day][curr_city]
                dp[day][curr_city] = max(dp[day][curr_city], stay_score)
                
                # Option 2: Travel from another city to current city
                for prev_city in range(n):
                    if prev_city != curr_city:
                        travel_score = dp[day-1][prev_city] + travelScore[prev_city][curr_city]
                        dp[day][curr_city] = max(dp[day][curr_city], travel_score)
        
        # Return maximum score achievable on the last day across all cities
        return max(dp[k-1])