from collections import deque

class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        dp = [0] * n
        dp[0] = energy[0]
        dq = deque([0])
        
        for i in range(1, n):
            if dq[0] < i - k:
                dq.popleft()
            dp[i] = energy[i] + dp[dq[0]]
            while dq and dp[dq[-1]] <= dp[i]:
                dq.pop()
            dq.append(i)
        
        return max(dp)