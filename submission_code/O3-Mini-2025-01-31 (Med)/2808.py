from typing import List
import math

class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        # We need to assign some walls to the paid painter and the rest to the free painter.
        # The free painter can only work concurrently with the paid painter. Since the free painter
        # paints each wall in 1 unit of time, if the paid painter is busy for T time units then at most T
        # walls can be painted by the free painter.
        #
        # Suppose we choose a subset S of walls to be painted by the paid painter. Then:
        #   - The paid painter cost = sum(cost[i] for i in S)
        #   - The paid painter time = sum(time[i] for i in S)
        #   - The remaining walls = n - |S| must be painted by the free painter.
        # To allow this concurrently, we need:
        #       free painter capacity = sum(time[i] for i in S) >= (n - |S|).
        #
        # Rearranging:
        #       sum(time[i]) + |S| >= n.
        #
        # Notice that for each chosen wall i in S, we add (time[i] + 1) to this "capacity".
        # Thus, we need to pick a subset S such that:
        #       sum(time[i] + 1 for i in S) >= n,
        # while minimizing sum(cost[i] for i in S).
        #
        # Since n <= 500 (and each time[i] is at most 500), even if an individual (time[i] + 1)
        # might be up to 501, we only care about reaching capacity n.
        # 
        # We can solve this as a knapSack type DP where our "capacity" (target) is n.
        
        N = len(cost)
        INF = math.inf
        # dp[x] = min cost required to achieve a total capacity exactly = x (or at least, we cap at n)
        dp = [INF] * (n := len(cost)) # initial n is cost length but note we use target capacity defined by input n.
        # Since we need capacity from walls to be at least total n (which is the number of walls),
        # our dp array size will be (n + 1), where n is the original number of walls.
        dp = [INF] * (len(cost) + 1)  # but actually length(cost) is n; we want indices 0..n
        target = len(cost)
        dp = [INF] * (target + 1)
        dp[0] = 0
        
        # Process each wall as an "item" with "weight" = time[i] + 1 and cost = cost[i].
        for i in range(N):
            cap = time[i] + 1
            c = cost[i]
            # We do DP in reverse order to avoid reusing the same wall more than once
            new_dp = dp[:]  # copy of current dp values for update
            for curr in range(target + 1):
                if dp[curr] != INF:
                    nxt = curr + cap
                    if nxt > target:
                        nxt = target
                    # Relax the cost for new capacity level nxt
                    if dp[curr] + c < new_dp[nxt]:
                        new_dp[nxt] = dp[curr] + c
            dp = new_dp
            
        return dp[target]

# Below is a simple testing interface.
if __name__ == '__main__':
    # Sample tests as provided:
    sol = Solution()
    # Example 1:
    cost1 = [1, 2, 3, 2]
    time1 = [1, 2, 3, 2]
    print(sol.paintWalls(cost1, time1))  # Expected output: 3
    
    # Example 2:
    cost2 = [2, 3, 4, 2]
    time2 = [1, 1, 1, 1]
    print(sol.paintWalls(cost2, time2))  # Expected output: 4