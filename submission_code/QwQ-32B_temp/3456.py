from typing import List
from collections import defaultdict

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        INF = float('-inf')
        dp = defaultdict(lambda: [INF] * (k + 1))
        dp[None][0] = 0  # Initial state: last is None, transitions 0, length 0
        
        for x in nums:
            new_dp = defaultdict(lambda: [INF] * (k + 1))
            # Copy current dp to new_dp (not taking x)
            for last in dp:
                for c in range(k + 1):
                    new_dp[last][c] = dp[last][c]
            # Process taking x
            for prev_last in dp:
                for c in range(k + 1):
                    current_length = dp[prev_last][c]
                    if current_length == INF:
                        continue
                    # Compute new_c
                    if prev_last is None:
                        new_c = c
                    else:
                        new_c = c + (x != prev_last)
                    if new_c > k:
                        continue
                    new_length = current_length + 1
                    if new_length > new_dp[x][new_c]:
                        new_dp[x][new_c] = new_length
            dp = new_dp
        
        max_len = 0
        for last in dp:
            for c in range(k + 1):
                if dp[last][c] > max_len:
                    max_len = dp[last][c]
        return max_len