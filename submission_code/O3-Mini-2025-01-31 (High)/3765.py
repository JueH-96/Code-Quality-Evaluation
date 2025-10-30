from typing import List
import math

class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        INF = float('inf')
        # Precompute prefix sums.
        # pre_nums[i] = sum(nums[0...i-1])
        # pre_cost[i] = sum(cost[0...i-1])
        pre_nums = [0] * (n + 1)
        pre_cost = [0] * (n + 1)
        for i in range(1, n + 1):
            pre_nums[i] = pre_nums[i - 1] + nums[i - 1]
            pre_cost[i] = pre_cost[i - 1] + cost[i - 1]
        
        # dp[c][i]: minimum cost to partition the first i elements (indices 0..i-1) into exactly c segments.
        # We need dp[0][0] = 0 and dp[0][i>0] = INF (not possible).
        dp = [[INF] * (n + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        # Base case: using 1 segment (c == 1), we must take the entire array prefix as one subarray.
        # For i >= 1, the cost is:
        #   (sum(nums[0..i-1]) + k*1) * (sum(cost[0..i-1])) = (pre_nums[i] + k) * pre_cost[i]
        for i in range(1, n + 1):
            dp[1][i] = (pre_nums[i] + k) * pre_cost[i]
        
        # We now build dp[c] for c = 2 to n.
        # The recurrence (for i >= c) is:
        #   dp[c][i] = (pre_nums[i] + k*c)*pre_cost[i] + min_{j from (c-1) to (i-1)} [dp[c-1][j] - (pre_nums[i] + k*c)*pre_cost[j]]
        #
        # Notice that for a fixed c and for each i, we need:
        #   min_{j in [c-1, i-1]} { dp[c-1][j] - (pre_nums[i] + k*c) * pre_cost[j] }
        # Let x = pre_nums[i] + k*c. Then for each valid candidate j (which is less than i),
        # we consider the value: dp[c-1][j] - pre_cost[j] * x.
        #
        # This fits the convex hull trick framework where each candidate j gives a line:
        #    y = m * x + b   with   m = -pre_cost[j]  and b = dp[c-1][j],
        # and we want to query for the minimum y at x = pre_nums[i] + k*c.
        #
        # Since cost values are positive, pre_cost is strictly increasing and so
        # the slopes m = -pre_cost[j] are (strictly) decreasing as j increases.
        # Furthermore, because pre_nums is increasing, x = pre_nums[i] + k*c is also increasing in i.
        # Thus we can use an efficient convex hull trick.
        
        # Define a helper convex hull class for minimum queries.
        class ConvexHull:
            def __init__(self):
                # Each element is a list [m, b, x_left],
                # where the line is y = m*x + b and x_left is the first x-value at which this line becomes optimal.
                self.hull = []
                self.ptr = 0  # pointer for online querying (since x queries are non-decreasing)
            
            def add_line(self, m, b):
                # New line: y = m*x + b.
                # Remove last lines if they are not needed.
                new_line = [m, b, -math.inf]
                while self.hull:
                    last = self.hull[-1]
                    # Compute x-coordinate of intersection between new_line and last.
                    # Solve: last[0]*x + last[1] = m*x + b  ==>  x = (b - last[1]) / (last[0] - m)
                    inter = (new_line[1] - last[1]) / (last[0] - new_line[0])
                    if inter <= last[2]:
                        self.hull.pop()
                    else:
                        new_line[2] = inter
                        break
                self.hull.append(new_line)
            
            def query(self, x):
                # x queries are made in non-decreasing order, so we can advance the pointer.
                if self.ptr >= len(self.hull):
                    self.ptr = len(self.hull) - 1
                while self.ptr < len(self.hull) - 1 and self.hull[self.ptr + 1][2] <= x:
                    self.ptr += 1
                m, b, _ = self.hull[self.ptr]
                return m * x + b
        
        # Build dp for c = 2 to n.
        # For each c, we compute dp[c][i] for i=c,...,n.
        for c in range(2, n + 1):
            dp_prev = dp[c - 1]
            newdp = [INF] * (n + 1)
            # We'll use the convex hull to compute:
            #    newdp[i] = (pre_nums[i] + k*c) * pre_cost[i] + min_{j in [c-1, i-1]} (dp_prev[j] - (pre_nums[i] + k*c) * pre_cost[j])
            hull = ConvexHull()
            # The first candidate index is j = c - 1.
            hull.add_line(-pre_cost[c - 1], dp_prev[c - 1])
            # Loop over i from c to n (i is the number of elements used).
            for i in range(c, n + 1):
                x_val = pre_nums[i] + k * c
                best = hull.query(x_val)
                newdp[i] = x_val * pre_cost[i] + best
                # After processing i, add candidate j = i for future queries.
                if i < n:
                    hull.add_line(-pre_cost[i], dp_prev[i])
            dp[c] = newdp
        
        # The answer is the minimum total cost among all valid segmentations.
        # That is: answer = min{ dp[c][n] for c = 1 to n }.
        ans = INF
        for c in range(1, n + 1):
            if dp[c][n] < ans:
                ans = dp[c][n]
        return ans