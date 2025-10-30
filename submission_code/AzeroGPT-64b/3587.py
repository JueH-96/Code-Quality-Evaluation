from typing import List

class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # Initialize memoization array with -1s for values not yet calculated
        dp = [[-1 for _ in range(n)] for _ in range(k + 1)]
        
        def dfs(day, city):
            # Base case: if it's the last day, return the points earned by staying in the current city
            if day == k:
                return stayScore[day - 1][city]
            
            # Check if the result for the current state is already calculated
            if dp[day][city] != -1:
                return dp[day][city]
            
            # Option 1: Stay in the current city
            stay = stayScore[day][city] + dfs(day + 1, city)
            
            # Option 2: Travel to another city
            travel = max(travelScore[city][next_city] + dfs(day + 1, next_city) for next_city in range(n))
            
            # Choose the option that gives the maximum score
            dp[day][city] = max(stay, travel)
            
            return dp[day][city]
        
        # Find the maximum starting point by calculating the score for each city as the starting point
        max_points = max(dfs(0, starting_city) for starting_city in range(n))
        
        return max_points