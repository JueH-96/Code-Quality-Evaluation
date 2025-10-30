class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        dp = [[0] * n for _ in range(k)]  # dp[day][city]

        # Day 0 initialization
        for c in range(n):
            # Option to stay in the starting city c
            dp[0][c] = max(dp[0][c], stayScore[0][c])

        for c in range(n):
            # Option to move from starting city c to any other city
            for dest in range(n):
                if dest != c:
                    dp[0][dest] = max(dp[0][dest], travelScore[c][dest])

        # Iterate over days 1 to k-1
        for day in range(1, k):
            for prev_city in range(n):
                # Option to stay in the same city
                stay_points = dp[day - 1][prev_city] + stayScore[day][prev_city]
                dp[day][prev_city] = max(dp[day][prev_city], stay_points)

                # Option to move to another city
                for dest_city in range(n):
                    if dest_city != prev_city:
                        move_points = dp[day - 1][prev_city] + travelScore[prev_city][dest_city]
                        dp[day][dest_city] = max(dp[day][dest_city], move_points)

        # The answer is the maximum points on the last day across all cities
        return max(dp[k - 1])