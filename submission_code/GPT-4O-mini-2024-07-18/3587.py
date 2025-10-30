from typing import List

class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # dp[day][city] will store the maximum points we can earn by the end of 'day' in 'city'
        dp = [[0] * n for _ in range(k + 1)]
        
        # Initialize the first day with stay scores
        for city in range(n):
            dp[0][city] = stayScore[0][city]
        
        # Fill the dp table for each day
        for day in range(1, k + 1):
            for curr in range(n):
                # If we stay in the current city
                dp[day][curr] = dp[day - 1][curr] + stayScore[day][curr]
                
                # If we move to another city
                for dest in range(n):
                    if dest != curr:
                        dp[day][dest] = max(dp[day][dest], dp[day - 1][curr] + travelScore[curr][dest])
        
        # The answer is the maximum points we can earn by the end of day k in any city
        return max(dp[k])