from collections import deque
from typing import List

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        dp = [0] * n
        lines = deque()  # each element is a tuple (m, b)
        
        # Add the first line (j=0)
        m0 = nums[0]
        b0 = dp[0] - 0 * nums[0]
        lines.append((m0, b0))
        
        for i in range(1, n):
            # Query the maximum value at x = i
            while len(lines) >= 2:
                m1, b1 = lines[0]
                m2, b2 = lines[1]
                val1 = m1 * i + b1
                val2 = m2 * i + b2
                if val2 > val1:
                    lines.popleft()
                else:
                    break
            m, b = lines[0]
            dp[i] = m * i + b
            
            # Add the new line for j = i
            new_m = nums[i]
            new_b = dp[i] - i * new_m
            new_line = (new_m, new_b)
            
            # Maintain the deque by removing lines that are no longer useful
            while len(lines) >= 2:
                m2, b2 = lines[-2]
                m1, b1 = lines[-1]
                # Compute lhs and rhs to determine if the last line can be removed
                lhs = (new_b - b1) * (m2 - m1)
                rhs = (b1 - b2) * (m1 - new_m)
                if lhs <= rhs:
                    lines.pop()
                else:
                    break
            lines.append(new_line)
        
        return dp[-1]