from collections import deque
from typing import List

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        dp = [0] * n
        dq = deque()
        
        # Initialize the deque with the line for index 0
        m = nums[0]
        b = dp[0] - 0 * m
        dq.append((m, b))
        
        for i in range(1, n):
            # Query the deque to find the maximum value at x = i
            while len(dq) >= 2:
                l1 = dq[0]
                l2 = dq[1]
                val1 = l1[0] * i + l1[1]
                val2 = l2[0] * i + l2[1]
                if val2 > val1:
                    dq.popleft()
                else:
                    break
            
            # Calculate the maximum score for the current index
            if dq:
                dp[i] = dq[0][0] * i + dq[0][1]
            else:
                dp[i] = 0  # This case should not occur as we start with at least one line
            
            # Add the new line for the current index
            m_new = nums[i]
            b_new = dp[i] - i * m_new
            
            # Maintain the deque by removing lines that are no longer useful
            while dq and m_new > dq[-1][0] and (m_new * i + b_new) > (dq[-1][0] * i + dq[-1][1]):
                dq.pop()
            
            dq.append((m_new, b_new))
        
        return dp[-1]