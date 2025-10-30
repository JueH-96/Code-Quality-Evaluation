class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # Initialize DP table for day 0
        dp = [stayScore[0][city] for city in range(n)]
        
        # Fill the DP table for days 1 to k-1
        for day in range(1, k):
            new_dp = [0] * n
            for curr_city in range(n):
                # Option 1: Stay in the current city
                new_dp[curr_city] = dp[curr_city] + stayScore[day][curr_city]
                
                # Option 2: Move from another city
                for prev_city in range(n):
                    if prev_city != curr_city:
                        new_dp[curr_city] = max(new_dp[curr_city], 
                                               dp[prev_city] + travelScore[prev_city][curr_city])
            
            dp = new_dp
        
        # Return the maximum value from the last day
        return max(dp)