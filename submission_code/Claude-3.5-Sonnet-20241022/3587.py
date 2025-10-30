class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # Create a DP table: dp[day][city] represents max score possible starting at day 'day' in city 'city'
        dp = {}
        
        def solve(day, curr_city):
            # Base case: if we've used all days, return 0
            if day == k:
                return 0
            
            # If this state has been computed before, return cached result
            if (day, curr_city) in dp:
                return dp[(day, curr_city)]
            
            max_points = 0
            
            # Option 1: Stay in current city
            stay_points = stayScore[day][curr_city] + solve(day + 1, curr_city)
            max_points = max(max_points, stay_points)
            
            # Option 2: Travel to another city
            for next_city in range(n):
                if next_city != curr_city:
                    travel_points = travelScore[curr_city][next_city] + solve(day + 1, next_city)
                    max_points = max(max_points, travel_points)
            
            # Cache and return result
            dp[(day, curr_city)] = max_points
            return max_points
        
        # Try starting from each city and find maximum possible score
        max_score = 0
        for start_city in range(n):
            max_score = max(max_score, solve(0, start_city))
            
        return max_score