import sys
from collections import deque
from typing import List

class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        INF = float('inf')
        
        # Compute prefix sums for nums and cost
        prefix_sum = [0] * (n + 1)
        prefix_cost = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + nums[i]
            prefix_cost[i+1] = prefix_cost[i] + cost[i]
        
        # Initialize DP table
        dp = [[INF] * (n + 1) for _ in range(n + 1)]
        dp[0][0] = 0  # Base case: 0 elements, 0 subarrays
        
        for m in range(1, n + 1):
            dq = deque()  # deque to maintain the convex hull lines
            for i in range(1, n + 1):
                j = i - 1
                # The line parameters for j (slope and intercept)
                slope = -prefix_cost[j]
                intercept = dp[j][m-1]
                
                # Maintain the deque to keep only useful lines
                while len(dq) >= 2:
                    m1, b1 = dq[-2]
                    m2, b2 = dq[-1]
                    # Compute x1 where the last two lines intersect
                    x1 = (b2 - b1) / (m1 - m2)
                    # Compute x2 where the new line intersects with the last line
                    x2 = (intercept - b2) / (m2 - slope)
                    if x2 <= x1:
                        dq.pop()
                    else:
                        break
                dq.append((slope, intercept))
                
                # Calculate C for current i and m
                C = prefix_sum[i] + k * m
                # Query the deque to find the best line for current C
                while len(dq) >= 2:
                    m0, b0 = dq[0]
                    m1, b1 = dq[1]
                    x_intersect = (b1 - b0) / (m0 - m1)
                    if C >= x_intersect:
                        dq.popleft()
                    else:
                        break
                # The first line in deque is the best
                m0, b0 = dq[0]
                min_val = m0 * C + b0
                dp[i][m] = C * prefix_cost[i] + min_val
        
        # The answer is the minimum over all possible m (1 to n)
        return min(dp[n][m] for m in range(1, n + 1))