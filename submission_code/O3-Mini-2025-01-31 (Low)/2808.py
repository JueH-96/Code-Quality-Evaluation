from typing import List
import math

class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        n = len(cost)
        # Our observation:
        # Let S be the set of walls painted by the paid painter. For each wall i in S,
        # paid painter takes time[i] units and free painter paints each wall in 1 unit.
        # We have free painter available only while the paid painter is working.
        # Since free painter paints sequentially (one per unit time), in the paid painter's
        # busy period of T = sum(time[i] for i in S), he can cover at most T walls.
        # The total walls painted: |S| (paid) + free painter walls.
        # But we must cover all n walls so, if S is the set of walls painted by the paid painter,
        # then free painter paints the remaining n - |S| walls.
        # In order for the free painter to be fully busy concurrently while paid painter works, we must have:
        #    T >= n - |S|
        # Rearranging: T + |S| >= n.
        #
        # Thus, for any chosen set S (the paid ones) with total time T and count S_count,
        # the constraint becomes:
        #    (sum(time[i] for i in S) + |S|) >= n.
        # Our goal is to choose a set S that minimizes the total cost, sum(cost[i] for i in S), while
        # satisfying the constraint above.
        #
        # We use a knapSack-like DP: Let's define dp[x] as the minimum cost to achieve a sum
        # of (time + 1) equal to exactly x. For each wall i, picking it contributes (time[i] + 1) to this sum,
        # at a cost of cost[i].
        #
        # Then the answer is min{ dp[x] } for all x >= n.
        
        # Determine maximum possible x we need to consider.
        # In worst-case, we might pick all walls. Each wall contributes at most 501 (since time[i] <= 500).
        # So maximum sum = n * 501.
        max_sum = n * 501
        
        # Initialize dp array: dp[x] = minimum cost to reach sum x.
        dp = [math.inf] * (max_sum + 1)
        dp[0] = 0
        
        for i in range(n):
            add_value = time[i] + 1
            c = cost[i]
            # Iterate backwards to avoid double counting the same wall
            for x in range(max_sum - add_value, -1, -1):
                if dp[x] != math.inf:
                    if dp[x] + c < dp[x + add_value]:
                        dp[x + add_value] = dp[x] + c
        
        # Now look for the minimum cost for all sums x >= n
        answer = min(dp[n:])
        return answer