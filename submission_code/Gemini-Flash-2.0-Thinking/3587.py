class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        dp = [[float('-inf')] * n for _ in range(k)]

        # Base case: end of day 0
        for current_city in range(n):
            dp[0][current_city] = stayScore[0][current_city]
            for prev_city in range(n):
                if prev_city != current_city:
                    dp[0][current_city] = max(dp[0][current_city], travelScore[prev_city][current_city])

        # Iterate through days
        for day in range(1, k):
            for current_city in range(n):
                # Stay in the current city
                dp[day][current_city] = max(dp[day][current_city], dp[day - 1][current_city] + stayScore[day][current_city])

                # Travel to the current city
                for prev_city in range(n):
                    if prev_city != current_city:
                        dp[day][current_city] = max(dp[day][current_city], dp[day - 1][prev_city] + travelScore[prev_city][current_city])

        return max(dp[k - 1]) if k > 0 else 0