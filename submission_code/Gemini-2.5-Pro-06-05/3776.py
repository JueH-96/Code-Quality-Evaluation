import sys
from typing import List

class Solution:
    def minCost(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Base cases for small n
        if n == 0:
            return 0
        if n == 1:
            return nums[0]
        if n == 2:
            return max(nums[0], nums[1])

        # dp0[i]: min cost to remove nums[i:]
        # We add padding to simplify boundary checks inside the loop.
        dp0 = [0] * (n + 4)
        
        # dp1[i][j]: min cost to remove [nums[j]] + nums[i:] (where j < i-1)
        dp1 = [[0] * n for _ in range(n + 4)]

        # Iterate from the end of the array to the beginning.
        for i in range(n, -1, -1):
            # --- Calculate dp0[i] ---
            if i >= n - 2:
                if i == n:
                    dp0[i] = 0
                elif i == n - 1:
                    dp0[i] = nums[i]
                else: # i == n-2
                    dp0[i] = max(nums[i], nums[i+1])
            else: # i < n-2
                a, b, c = nums[i], nums[i+1], nums[i+2]
                
                # Option 1: remove a, b. Survivor c. -> list is nums[i+2:]
                c1 = max(a, b) + dp0[i+2]
                
                # Option 2: remove a, c. Survivor b. -> list is [b] + nums[i+3:]
                c2 = max(a, c) + dp1[i+3][i+1]
                
                # Option 3: remove b, c. Survivor a. -> list is [a] + nums[i+3:]
                c3 = max(b, c) + dp1[i+3][i]
                
                dp0[i] = min(c1, c2, c3)

            # --- Calculate dp1[i][j] for all j < i-1 ---
            for j in range(i - 2, -1, -1):
                val = nums[j]
                
                if i >= n - 1:
                    if i == n: # list is [val]
                        dp1[i][j] = val
                    else: # i == n-1, list is [val, nums[n-1]]
                        dp1[i][j] = max(val, nums[i])
                    continue
                
                # i < n-1, list is [val, nums[i], nums[i+1], ...]
                a, b = nums[i], nums[i+1]
                
                # Option 1: remove val, a. Survivor b. -> list is nums[i+1:]
                c1 = max(val, a) + dp0[i+1]
                
                # Option 2: remove val, b. Survivor a. -> list is [a] + nums[i+2:]
                c2 = max(val, b) + dp1[i+2][i]
                
                # Option 3: remove a, b. Survivor val. -> list is [val] + nums[i+2:]
                c3 = max(a, b) + dp1[i+2][j]
                
                dp1[i][j] = min(c1, c2, c3)
            
        return dp0[0]