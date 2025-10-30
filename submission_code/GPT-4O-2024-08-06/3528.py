from typing import List
from collections import deque

class Solution:
    def findMaximumScore(self, nums: List[int]) -> int:
        n = len(nums)
        # dp[i] will store the maximum score to reach index i
        dp = [0] * n
        # We start at index 0, so the score is 0 initially
        dp[0] = 0
        
        # We will use a deque to keep track of the indices with the best scores
        # The deque will store indices in decreasing order of their dp values
        dq = deque([0])
        
        for i in range(1, n):
            # Remove indices from the deque that are not reachable from the current index
            while dq and dq[0] < i - 1:
                dq.popleft()
            
            # Calculate the score for jumping from the best index in the deque to the current index
            dp[i] = dp[dq[0]] + (i - dq[0]) * nums[dq[0]]
            
            # Maintain the deque in decreasing order of dp values
            while dq and dp[dq[-1]] <= dp[i]:
                dq.pop()
            
            # Add the current index to the deque
            dq.append(i)
        
        # The answer is the maximum score to reach the last index
        return dp[-1]