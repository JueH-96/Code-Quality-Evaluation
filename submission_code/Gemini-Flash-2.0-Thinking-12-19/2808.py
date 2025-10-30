class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        dp = {} # (index, walls_painted_by_paid, required_time) -> min_cost

        def solve(index, walls_painted_by_paid, required_time):
            if required_time <= 0:
                return 0
            if index == n:
                return float('inf')
            if (index, walls_painted_by_paid, required_time) in dp:
                return dp[(index, walls_painted_by_paid, required_time)]

            # Option 1: Use paid painter for wall at index
            cost_paid_painter = cost[index] + solve(index + 1, walls_painted_by_paid + 1, max(0, required_time - time[index] - 1))

            # Option 2: Use free painter for wall at index (or skip using paid for this wall and let free painter do it if possible)
            cost_free_painter = solve(index + 1, walls_painted_by_paid, max(0, required_time - 1))

            result = min(cost_paid_painter, cost_free_painter)
            dp[(index, walls_painted_by_paid, required_time)] = result
            return result

        min_total_cost = solve(0, 0, n)
        return min_total_cost