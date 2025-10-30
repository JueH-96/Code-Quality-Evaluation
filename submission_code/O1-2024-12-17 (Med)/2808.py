class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        import math

        n = len(cost)
        # dp[used][covered] = minimum total cost when we have chosen `used` walls to pay for
        # and thereby covered (in parallel) `covered` walls' worth of free-painter time.
        # The free painter can paint up to `covered` walls (one per time unit),
        # while the paid painter is busy with the `used` chosen walls.
        # We need used + covered >= n to paint all walls.
        
        INF = math.inf
        dp = [[INF]*(n+1) for _ in range(n+1)]
        dp[0][0] = 0  # No walls chosen, no coverage, cost = 0
        
        for i in range(n):
            # Paint wall i if paid
            wall_cost = cost[i]
            wall_time = time[i]
            # Traverse backwards so that we do not overwrite states we still need to process
            for used in range(n-1, -1, -1):
                for covered in range(n+1):
                    if dp[used][covered] == INF:
                        continue
                    new_covered = min(n, covered + wall_time)
                    dp[used+1][new_covered] = min(dp[used+1][new_covered],
                                                  dp[used][covered] + wall_cost)
        
        # Find the minimal cost among all (used, covered) where used + covered >= n
        ans = INF
        for used in range(n+1):
            for covered in range(n+1):
                if used + covered >= n:
                    ans = min(ans, dp[used][covered])
        
        return ans