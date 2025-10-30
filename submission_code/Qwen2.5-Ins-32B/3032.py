from typing import List

class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        max_val = 0
        
        # Precompute the sum of ids for each starting player up to k passes
        dp = [[0] * 40 for _ in range(n)]
        for i in range(n):
            dp[i][0] = i
        for j in range(1, 40):
            for i in range(n):
                dp[i][j] = dp[i][j-1] + dp[receiver[i]][j-1]
        
        # Calculate the maximum value of the function
        for i in range(n):
            total = i
            x = i
            for j in range(39, -1, -1):
                if (k >> j) & 1:
                    total += dp[x][j]
                    x = receiver[x]
            max_val = max(max_val, total)
        
        return max_val