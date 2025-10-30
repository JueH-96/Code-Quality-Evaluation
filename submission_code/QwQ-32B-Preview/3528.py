from typing import List
from collections import deque

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0  # Already at the last index
        
        # Initialize dp array
        dp = [0] * n
        
        # Deque to maintain the lines in the convex hull
        dq = deque()
        
        # Add the last index line to the deque
        # Line: y = dp[n-1] + n-1 * x = 0 + (n-1)*x
        dq.append((0, n-1))
        
        # Iterate from n-2 down to 0
        for i in range(n-2, -1, -1):
            # Query the maximum value at x = nums[i]
            a, b = dq[0]
            dp[i] = a + b * nums[i] - i * nums[i]
            
            # Add the new line y = dp[i] + i * x
            new_line = (dp[i], i)
            # Remove lines from the end that are dominated by the new line
            while len(dq) >= 2:
                a1, b1 = dq[-2]
                a2, b2 = dq[-1]
                a3, b3 = new_line
                x1 = (a2 - a1) / (b1 - b2) if b1 != b2 else float('inf')
                x2 = (a3 - a2) / (b2 - b3) if b2 != b3 else float('inf')
                if x1 < x2:
                    break
                dq.pop()
            dq.append(new_line)
        
        return dp[0]