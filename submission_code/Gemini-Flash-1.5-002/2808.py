class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        dp = {}

        def solve(i, paid_time):
            if i == n:
                return 0
            if (i, paid_time) in dp:
                return dp[(i, paid_time)]

            # Option 1: Paid painter paints wall i
            paid_cost = cost[i]
            paid_time_taken = time[i]
            free_time = paid_time_taken
            
            min_cost = paid_cost + solve(i + 1, paid_time + paid_time_taken)


            # Option 2: Free painter paints wall i (if possible)
            if paid_time >= 1:
                free_cost = 0
                min_cost = min(min_cost, free_cost + solve(i + 1, paid_time + 1))

            dp[(i, paid_time)] = min_cost
            return min_cost

        return solve(0, 0)