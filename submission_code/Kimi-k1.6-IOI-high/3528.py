from typing import List
from collections import deque

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        
        dp = [0] * n
        dp[0] = 0
        
        dq = deque()
        a0 = nums[0]
        b0 = dp[0] - 0 * nums[0]
        dq.append((a0, b0))
        
        for i in range(1, n):
            while len(dq) >= 2:
                a1, b1 = dq[0]
                a2, b2 = dq[1]
                if a1 * i + b1 < a2 * i + b2:
                    dq.popleft()
                else:
                    break
            a, b = dq[0]
            dp[i] = a * i + b
            
            new_a = nums[i]
            new_b = dp[i] - i * new_a
            while len(dq) >= 2:
                a_pp, b_pp = dq[-2]
                a_p, b_p = dq[-1]
                determinant = (new_b - b_pp) * (a_pp - a_p) - (b_p - b_pp) * (a_pp - new_a)
                if determinant <= 0:
                    dq.pop()
                else:
                    break
            dq.append((new_a, new_b))
        
        return dp[-1]