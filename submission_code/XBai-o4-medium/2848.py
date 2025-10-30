from typing import List

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 0:
            return 0
        
        # Initialize DP table: dp[mask][i] represents the number of ways to reach mask ending with i-th element
        dp = [[0] * n for _ in range(1 << n)]
        
        # Base case: masks with single element
        for i in range(n):
            dp[1 << i][i] = 1
        
        # Iterate over all possible masks
        for mask in range(1, 1 << n):
            for last in range(n):
                # Check if last element is in the current mask
                if not (mask & (1 << last)):
                    continue
                # Try to add a new element 'next_last' not in the mask
                for next_last in range(n):
                    if mask & (1 << next_last):
                        continue
                    # Check compatibility between last and next_last
                    if (nums[last] % nums[next_last] == 0) or (nums[next_last] % nums[last] == 0):
                        new_mask = mask | (1 << next_last)
                        dp[new_mask][next_last] = (dp[new_mask][next_last] + dp[mask][last]) % MOD
        
        # Full mask is all bits set, sum all possible endings
        full_mask = (1 << n) - 1
        return sum(dp[full_mask][i] for i in range(n)) % MOD