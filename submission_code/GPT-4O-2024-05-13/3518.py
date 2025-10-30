from typing import List
from heapq import heappush, heappop

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        dp = [[-float('inf')] * 5 for _ in range(n)]
        
        for i in range(n):
            dp[i][0] = 0
        
        for i in range(n):
            for j in range(1, 5):
                if i > 0:
                    dp[i][j] = dp[i-1][j]
                if i >= j:
                    dp[i][j] = max(dp[i][j], dp[i-1][j-1] + a[j-1] * b[i])
        
        return dp[n-1][4]

# Example usage:
# sol = Solution()
# print(sol.maxScore([3,2,5,6], [2,-6,4,-5,-3,2,-7]))  # Output: 26
# print(sol.maxScore([-1,4,5,-2], [-5,-1,-3,-2,-4]))  # Output: -1