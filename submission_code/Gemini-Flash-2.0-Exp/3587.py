class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        dp = [[0] * n for _ in range(k + 1)]
        
        # Initialize the base case for day 0
        for city in range(n):
            dp[0][city] = 0  # Starting with 0 points

        # Iterate through each day
        for day in range(1, k + 1):
            # Iterate through each city
            for curr_city in range(n):
                # Option 1: Stay in the current city
                dp[day][curr_city] = dp[day - 1][curr_city] + stayScore[day - 1][curr_city]

                # Option 2: Travel from another city to the current city
                for prev_city in range(n):
                    if prev_city != curr_city:
                        dp[day][curr_city] = max(dp[day][curr_city], dp[day - 1][prev_city] + travelScore[prev_city][curr_city])

        # Find the maximum score among all cities on the last day
        max_score = 0
        for city in range(n):
            max_score = max(max_score, dp[k][city])

        return max_score