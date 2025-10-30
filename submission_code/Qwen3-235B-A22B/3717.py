from typing import List

class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        import sys
        input = sys.stdin.read
        n = len(nums)
        m = n - x + 1
        if m < k:
            return -1  # Should not happen per constraints
        
        # Precompute the cost for each block
        cost = []
        for i in range(m):
            block = nums[i:i+x]
            block.sort()
            median = block[x // 2]
            total = 0
            for num in block:
                total += abs(num - median)
            cost.append(total)
        
        INF = float('inf')
        dp = [[INF] * (k + 1) for _ in range(m)]
        # Initialize dp for j=0 (selecting 0 blocks)
        for i in range(m):
            dp[i][0] = 0
        
        # prefix_min[j][i] is the minimal dp[j][p] for p <= i
        prefix_min = [[INF] * m for _ in range(k + 1)]
        # Initialize prefix_min for j=0
        for i in range(m):
            prefix_min[0][i] = 0
        
        for i in range(m):
            # Fill dp for current block i
            for j in range(1, k + 1):
                # Option 1: Do not take block i
                option1 = dp[i-1][j] if i > 0 else INF
                # Option 2: Take block i
                option2 = INF
                prev_pos = i - x
                if prev_pos >= 0:
                    prev_min = prefix_min[j-1][prev_pos]
                    if prev_min != INF:
                        option2 = prev_min + cost[i]
                else:
                    if j == 1:
                        option2 = cost[i]
                dp[i][j] = min(option1, option2)
            # Update prefix_min for each j
            for j in range(k + 1):
                if j == 0:
                    prefix_min[j][i] = 0
                else:
                    if i == 0:
                        prefix_min[j][i] = dp[i][j]
                    else:
                        prefix_min[j][i] = min(prefix_min[j][i-1], dp[i][j])
        
        return dp[m-1][k] if dp[m-1][k] != INF else -1