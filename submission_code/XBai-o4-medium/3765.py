from typing import List
from collections import deque

class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        pre_sum = [0] * (n + 1)
        for i in range(1, n + 1):
            pre_sum[i] = pre_sum[i - 1] + nums[i - 1]
        
        cost_pre = [0] * (n + 1)
        for i in range(1, n + 1):
            cost_pre[i] = cost_pre[i - 1] + cost[i - 1]
        
        INF = float('inf')
        dp = [[INF] * (n + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        # Initialize for j=1
        for i in range(1, n + 1):
            dp[i][1] = (pre_sum[i] + k) * cost_pre[i]
        
        # Process j >= 2
        for j in range(2, n + 1):
            dq = deque()
            for i in range(j, n + 1):
                m_candidate = i - 1
                slope = -cost_pre[m_candidate]
                intercept = dp[m_candidate][j - 1]
                
                # Add the new line to deque, maintaining convex hull
                while len(dq) >= 2:
                    # Get last two lines in deque
                    l1_m, l1_b = dq[-1]
                    l2_m, l2_b = dq[-2]
                    # Compute x1: intersection between l2 and l1
                    x1_num = (l1_b - l2_b)
                    x1_den = (l2_m - l1_m)
                    # Compute x2: intersection between l1 and new line
                    x2_num = (intercept - l1_b)
                    x2_den = (l1_m - slope)
                    # Check if x2 <= x1
                    if x1_num * x2_den >= x2_num * x1_den:
                        dq.pop()
                    else:
                        break
                dq.append((slope, intercept))
                
                # Compute A for current i and j
                A = pre_sum[i] + k * j
                x = A
                
                # Query the deque for the best line
                while len(dq) >= 2:
                    l0_m, l0_b = dq[0]
                    l1_m, l1_b = dq[1]
                    if l0_m * x + l0_b >= l1_m * x + l1_b:
                        dq.popleft()
                    else:
                        break
                best_val = dq[0][0] * x + dq[0][1]
                dp[i][j] = A * cost_pre[i] + best_val
        
        # Find the minimum total cost among all possible number of subarrays
        res = INF
        for j in range(1, n + 1):
            if dp[n][j] < res:
                res = dp[n][j]
        return res