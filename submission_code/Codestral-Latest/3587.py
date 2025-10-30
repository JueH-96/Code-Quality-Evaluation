class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # Initialize a 2D DP array where dp[i][j] represents the maximum points
        # that can be earned on day i when the tourist is in city j.
        dp = [[0] * n for _ in range(k)]

        # Initialize the first day's points based on staying in each city.
        for city in range(n):
            dp[0][city] = stayScore[0][city]

        # Fill the DP table for each day and each city.
        for day in range(1, k):
            for curr_city in range(n):
                # Option 1: Stay in the current city.
                stay_points = dp[day - 1][curr_city] + stayScore[day][curr_city]

                # Option 2: Move to another city.
                move_points = 0
                for dest_city in range(n):
                    if curr_city != dest_city:
                        move_points = max(move_points, dp[day - 1][dest_city] + travelScore[dest_city][curr_city])

                # The maximum points for the current day and city is the maximum of staying or moving.
                dp[day][curr_city] = max(stay_points, move_points)

        # The result is the maximum value in the last day's row of the DP table.
        return max(dp[k - 1])