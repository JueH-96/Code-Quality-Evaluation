import collections
from typing import List

class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        A = [0] * n
        B = [0] * n
        A[0] = nums[0]
        B[0] = cost[0]
        for i in range(1, n):
            A[i] = A[i-1] + nums[i]
            B[i] = B[i-1] + cost[i]
        
        dp_prev = [0] * n
        for i in range(n):
            dp_prev[i] = (A[i] + k) * B[i]
        ans = dp_prev[n-1]
        
        for s_val in range(2, n+1):
            dp_curr = [0] * n
            dq = collections.deque()
            for i in range(s_val-1, n):
                j = i - 1
                m_new = -B[j]
                c_new = dp_prev[j] - k * s_val * B[j]
                
                while len(dq) >= 2:
                    m1, c1 = dq[-2]
                    m2, c2 = dq[-1]
                    if (c_new - c1) * (m1 - m2) <= (c2 - c1) * (m1 - m_new):
                        dq.pop()
                    else:
                        break
                dq.append((m_new, c_new))
                
                while len(dq) >= 2:
                    m1, c1 = dq[0]
                    m2, c2 = dq[1]
                    if m1 * A[i] + c1 >= m2 * A[i] + c2:
                        dq.popleft()
                    else:
                        break
                min_val = dq[0][0] * A[i] + dq[0][1]
                dp_curr[i] = (A[i] + k * s_val) * B[i] + min_val
            
            ans = min(ans, dp_curr[n-1])
            dp_prev = dp_curr
        
        return ans