from typing import List
from collections import deque

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        dq = deque()
        dq.append((nums[0], 0))
        dp = [0] * n
        for i in range(1, n):
            # Optimize the front of the deque
            while len(dq) >= 2:
                m1, b1 = dq[0]
                m2, b2 = dq[1]
                if m2 * i + b2 >= m1 * i + b1:
                    dq.popleft()
                else:
                    break
            # Get the best value for current i
            m_best, b_best = dq[0]
            dp[i] = m_best * i + b_best
            # Create the new line for current i
            m_new = nums[i]
            b_new = dp[i] - m_new * i
            # Add the new line to the deque, maintaining the invariant
            while dq:
                m_last, b_last = dq[-1]
                if m_new > m_last:
                    numerator = b_last - b_new
                    denominator = m_new - m_last
                    x_intersect = numerator / denominator
                    if x_intersect <= i:
                        dq.pop()
                    else:
                        break
                else:
                    break
            dq.append((m_new, b_new))
        return dp[-1]