class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # Initialize dp array with dimensions [k+1][n]
        dp = [[0] * n for _ in range(k + 1)]
        
        # Base case: fill in the scores for the first day (day 0)
        for city in range(n):
            dp[0][city] = stayScore[0][city]
        
        # Iterate through each day
        for day in range(1, k):
            # For each city as current position
            for curr_city in range(n):
                # Option 1: Stay in the current city
                stay_score = dp[day-1][curr_city] + stayScore[day][curr_city]
                
                # Option 2: Travel from any other city
                travel_scores = [dp[day-1][prev_city] + travelScore[prev_city][curr_city] 
                                 for prev_city in range(n) if prev_city != curr_city]
                
                # Take the maximum of staying or traveling
                dp[day][curr_city] = max(stay_score, max(travel_scores))
        
        # Return the maximum score achievable on the last day
        return max(dp[k-1])