class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        import math

        n = len(cost)
        # dp[i][j] = minimum total cost using first i walls (0..i-1) to achieve a "coverage" of j
        # "coverage" = sum_of_chosen_time + number_of_chosen_walls
        # We want to achieve coverage >= n (so that the free painter can paint the rest).
        dp = [[math.inf] * (n + 1) for _ in range(n + 1)]
        dp[0][0] = 0  # No walls chosen => cost=0, coverage=0

        for i in range(1, n + 1):
            t = time[i - 1]
            c = cost[i - 1]
            # We can either skip wall i-1 (use free painter) or pay for it
            for j in range(n + 1):
                # Skip wall i-1
                dp[i][j] = min(dp[i][j], dp[i - 1][j])
                # Pay for wall i-1
                # coverage gained = t + 1
                new_cover = min(n, j + t + 1)
                dp[i][new_cover] = min(dp[i][new_cover], dp[i - 1][j] + c)

        return dp[n][n]