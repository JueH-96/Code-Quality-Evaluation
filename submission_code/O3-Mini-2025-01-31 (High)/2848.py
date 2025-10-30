from typing import List

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        n = len(nums)
        mod = 10**9 + 7
        
        # Precompute allowed transitions.
        # allowed[i][j] is True if nums[i] and nums[j] can be adjacent.
        allowed = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0:
                    allowed[i][j] = True
        
        # dp[mask][i]: number of valid ways to form permutations using the subset 'mask'
        # that end with the element at index i.
        dp = [[0] * n for _ in range(1 << n)]
        
        # Initialize for single element permutations.
        for i in range(n):
            dp[1 << i][i] = 1
        
        # Iterate over all subsets of indices.
        for mask in range(1 << n):
            for i in range(n):
                if dp[mask][i] != 0:
                    # Try to add an element j that is not yet used in mask.
                    for j in range(n):
                        if mask & (1 << j) == 0 and allowed[i][j]:
                            dp[mask | (1 << j)][j] = (dp[mask | (1 << j)][j] + dp[mask][i]) % mod
        
        # Sum ways to end with any element, with full mask.
        return sum(dp[(1 << n) - 1]) % mod