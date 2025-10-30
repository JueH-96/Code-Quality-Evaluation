class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        dp = {}

        def solve(i, remaining_walls):
            if remaining_walls <= 0:
                return 0
            if i == n:
                return float('inf')
            
            if (i, remaining_walls) in dp:
                return dp[(i, remaining_walls)]

            paint = cost[i] + solve(i + 1, remaining_walls - time[i] - 1)
            skip = solve(i + 1, remaining_walls)

            dp[(i, remaining_walls)] = min(paint, skip)
            return dp[(i, remaining_walls)]

        return solve(0, n)