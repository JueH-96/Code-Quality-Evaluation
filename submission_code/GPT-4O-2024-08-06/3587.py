class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # Initialize a dp array where dp[i][j] represents the maximum score on day i if the tourist is in city j
        dp = [[0] * n for _ in range(k)]
        
        # On the first day, the tourist can start in any city
        for city in range(n):
            dp[0][city] = stayScore[0][city]
        
        # Fill the dp table for each day
        for day in range(1, k):
            for curr_city in range(n):
                # Calculate the maximum score if staying in the current city
                max_stay = dp[day-1][curr_city] + stayScore[day][curr_city]
                
                # Calculate the maximum score if traveling to another city
                max_travel = 0
                for prev_city in range(n):
                    if prev_city != curr_city:
                        max_travel = max(max_travel, dp[day-1][prev_city] + travelScore[prev_city][curr_city])
                
                # The maximum score for this day and city is the best of staying or traveling
                dp[day][curr_city] = max(max_stay, max_travel)
        
        # The result is the maximum score on the last day in any city
        return max(dp[k-1])