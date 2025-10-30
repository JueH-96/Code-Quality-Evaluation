from typing import List

class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        # Initialize a dp array where dp[i][j] represents the maximum score possible
        # when the tourist is in city j on day i.
        dp = [[0] * n for _ in range(k)]

        # On the first day, the tourist can start in any city and earn the stay score for that city.
        for j in range(n):
            dp[0][j] = stayScore[0][j]

        # Fill the dp array for the remaining days.
        for i in range(1, k):
            for j in range(n):
                # Option 1: Stay in the same city.
                stay = dp[i-1][j] + stayScore[i][j]
                # Option 2: Move from another city.
                travel = max(dp[i-1][m] + travelScore[m][j] + stayScore[i][j] for m in range(n) if m != j)
                # Take the maximum of staying or traveling.
                dp[i][j] = max(stay, travel)

        # The answer will be the maximum score possible on the last day across all cities.
        return max(dp[k-1])