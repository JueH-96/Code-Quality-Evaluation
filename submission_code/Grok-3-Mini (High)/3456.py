import math
from typing import List
from collections import defaultdict

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        # Find the runs of consecutive equal elements
        runs = []
        i = 0
        while i < len(nums):
            val = nums[i]
            cnt = 0
            j = i
            while j < len(nums) and nums[j] == val:
                cnt += 1
                j += 1
            runs.append((val, cnt))
            i = j
        
        N = len(runs)
        if N == 0:
            return 0  # Though len nums >=1, safe to handle
        
        # dp[i][c]: max length ending with run i with exactly c changes, or -1 if impossible
        dp = [[-1 for _ in range(k + 1)] for _ in range(N)]
        
        # For c=0, need cumulative sum per value
        val_sum_dict = defaultdict(int)  # Cumulative sum before current run
        
        for i in range(N):
            val_i, size_i = runs[i]
            # Set dp[i][0]
            dp[i][0] = val_sum_dict[val_i] + size_i
            # Update the cumulative sum for val_i to include current run
            val_sum_dict[val_i] += size_i
            
            # Now set dp[i][c] for c=1 to k
            for c in range(1, k + 1):
                max_prev = -1  # Initialize to invalid
                for j in range(i):  # j from 0 to i-1
                    val_j = runs[j][0]
                    if val_j == val_i:
                        # Same value, delta=0, use dp[j][c]
                        if dp[j][c] >= 0:
                            if dp[j][c] > max_prev:
                                max_prev = dp[j][c]
                    else:  # Different value
                        if c - 1 >= 0 and dp[j][c - 1] >= 0:
                            if dp[j][c - 1] > max_prev:
                                max_prev = dp[j][c - 1]
                # After checking all j
                if max_prev >= 0:
                    dp[i][c] = max_prev + size_i
                else:
                    dp[i][c] = -1  # Impossible to have c changes ending here
        
        # Find the maximum length with at most k changes
        ans = 0
        for i in range(N):
            for c in range(k + 1):
                if dp[i][c] > ans:
                    ans = max(ans, dp[i][c])
        
        return ans