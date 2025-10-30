from typing import List
from collections import deque

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Step 1: Compute prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        # Step 2: Precompute coefficients
        c = [0] * (k + 1)
        for j in range(1, k + 1):
            c[j] = (-1)**(j + 1) * (k - j + 1)
        
        # Step 3: Initialize DP and deques
        dp = [[float('-inf')] * (k + 1) for _ in range(n)]
        deques = [deque() for _ in range(k + 1)]
        
        # Base case: dp[i][0] = 0 for all i
        for i in range(n):
            dp[i][0] = 0
        
        # Step 4: Iterate through each position i
        for i in range(n):
            # Update DP for each j
            for j in range(1, min(k, i + 1) + 1):
                if j > i + 1:
                    dp[i][j] = float('-inf')
                else:
                    M_j = deques[j][0] if deques[j] else float('-inf')
                    dp[i][j] = c[j] * (prefix[i + 1] - prefix[i - j + 2]) + M_j
            # Update deques for each j
            for j in range(1, min(k, i + 1) + 1):
                if j > i + 1:
                    continue
                val = dp[i][j - 1] - c[j] * prefix[i - j + 2]
                while deques[j] and val > deques[j][-1][0]:
                    deques[j].pop()
                deques[j].append((val, i))
        
        # Step 5: Return the result
        return dp[n - 1][k]