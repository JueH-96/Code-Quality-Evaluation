import collections
from typing import List

class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        P = [0] * n
        Q = [0] * n
        P[0] = nums[0]
        Q[0] = cost[0]
        for i in range(1, n):
            P[i] = P[i-1] + nums[i]
            Q[i] = Q[i-1] + cost[i]
        
        INF = 10**18
        dp = [[INF] * (n+1) for _ in range(n)]
        
        for i in range(n):
            dp[i][1] = (P[i] + k) * Q[i]
        
        if n == 1:
            return dp[0][1]
        
        for seg in range(2, n+1):
            dq = collections.deque()
            for i in range(seg-1, n):
                m_new = -Q[i-1]
                b_new = dp[i-1][seg-1] - k * seg * Q[i-1]
                
                while dq and dq[-1][0] == m_new and dq[-1][1] >= b_new:
                    dq.pop()
                if dq and dq[-1][0] == m_new and dq[-1][1] < b_new:
                    pass
                else:
                    while len(dq) >= 2:
                        m1, b1 = dq[-1]
                        m2, b2 = dq[-2]
                        left = (b_new - b2) * (m2 - m1)
                        right = (b1 - b2) * (m2 - m_new)
                        if left <= right:
                            dq.pop()
                        else:
                            break
                    dq.append((m_new, b_new))
                
                while len(dq) >= 2:
                    m1, b1 = dq[0]
                    m2, b2 = dq[1]
                    val1 = m1 * P[i] + b1
                    val2 = m2 * P[i] + b2
                    if val1 >= val2:
                        dq.popleft()
                    else:
                        break
                
                m1, b1 = dq[0]
                min_val = m1 * P[i] + b1
                total_cost = (P[i] + k * seg) * Q[i] + min_val
                if total_cost < dp[i][seg]:
                    dp[i][seg] = total_cost
        
        return min(dp[n-1][s] for s in range(1, n+1))