class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # Initialize a DP table where dp[i][j] represents the maximum score on day i, ending in city j
        dp = [[-1 for _ in range(n)] for _ in range(k)]
        
        # Initialize the first day
        for city in range(n):
            dp[0][city] = stayScore[0][city]
        
        # Fill the DP table for each day
        for day in range(1, k):
            for curr_city in range(n):
                max_score = -1
                # Option 1: Stay in the current city
                stay_option = dp[day-1][curr_city] + stayScore[day][curr_city]
                max_score = max(max_score, stay_option)
                # Option 2: Travel from any other city to the current city
                for prev_city in range(n):
                    if prev_city != curr_city:
                        travel_option = dp[day-1][prev_city] + travelScore[prev_city][curr_city]
                        max_score = max(max_score, travel_option)
                dp[day][curr_city] = max_score
        
        # The answer is the maximum value in the last day's DP row
        return max(dp[-1])