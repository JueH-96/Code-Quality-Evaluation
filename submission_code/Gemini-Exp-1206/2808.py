class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        dp = {}

        def solve(i, remain):
            if remain <= 0:
                return 0
            if i == n:
                return float('inf')
            if (i, remain) in dp:
                return dp[(i, remain)]

            paint = cost[i] + solve(i + 1, remain - 1 - time[i])
            not_paint = solve(i + 1, remain)
            dp[(i, remain)] = min(paint, not_paint)
            return dp[(i, remain)]

        return solve(0, n)