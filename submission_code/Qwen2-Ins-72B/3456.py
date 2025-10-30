from collections import deque

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [0] * (n + 1)
        q = deque()
        
        for i in range(n):
            while q and nums[q[-1]] == nums[i]:
                q.pop()
                
            if q:
                dp[i + 1] = dp[q[-1]] + (i - q[-1])
            else:
                dp[i + 1] = i + 1
            
            if dp[i + 1] <= k:
                q.append(i)
                
        return max(dp)