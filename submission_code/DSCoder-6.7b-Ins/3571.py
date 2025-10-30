from typing import List
from collections import defaultdict

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        coordinates.sort()
        n = len(coordinates)
        dp = [0] * n
        dp[k] = 1
        stack = [(coordinates[k][0], coordinates[k][1])]
        res = 1
        for i in range(k+1, n):
            while stack and (coordinates[i][0] <= stack[-1][0] or coordinates[i][1] <= stack[-1][1]):
                stack.pop()
            if stack:
                dp[i] = dp[stack[-1]] + 1
                res = max(res, dp[i])
            stack.append((coordinates[i][0], coordinates[i][1]))
        return res