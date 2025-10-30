class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # Create a DP table with (k+1) days and n cities
        dp = [[0 for _ in range(n)] for _ in range(k + 1)]
        
        # Iterate from day k-1 down to day 0
        for day in range(k - 1, -1, -1):
            for city in range(n):
                # Calculate the score if staying in the current city
                stay_score = stayScore[day][city] + dp[day + 1][city]
                # Initialize dp[day][city] with the stay score
                dp[day][city] = stay_score
                
                # Check all possible moves to other cities
                for dest in range(n):
                    if dest != city:
                        move_score = travelScore[city][dest] + dp[day + 1][dest]
                        # Update dp[day][city] if moving to dest is better
                        dp[day][city] = max(dp[day][city], move_score)
        
        # The answer is the maximum value in dp[0] over all starting cities
        return max(dp[0])