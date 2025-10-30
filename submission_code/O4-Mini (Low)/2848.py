from typing import List

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Precompute adjacency (special) matrix
        # adj[i][j] = True if nums[i] and nums[j] are divisible one way or the other
        adj = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    if nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0:
                        adj[i][j] = True
        
        # dp[mask][last] = number of ways to form a special permutation using the chosen bits in mask,
        # ending at index `last`
        full_mask = (1 << n)
        dp = [ [0]*n for _ in range(full_mask) ]
        
        # Initialize: single-element permutations
        for i in range(n):
            dp[1 << i][i] = 1
        
        # Fill DP
        for mask in range(full_mask):
            # Try to extend each permutation ending in `last`
            for last in range(n):
                count = dp[mask][last]
                if count:
                    # Try to add a new element `nxt` not yet used
                    free = (~mask) & (full_mask - 1)
                    # iterate through bits of free
                    m = free
                    while m:
                        nxt_bit = m & -m
                        m -= nxt_bit
                        nxt = (nxt_bit.bit_length() - 1)
                        if adj[last][nxt]:
                            dp[mask | nxt_bit][nxt] = (dp[mask | nxt_bit][nxt] + count) % MOD
        
        # Sum up all ways that use all elements
        res = 0
        final_mask = full_mask - 1
        for i in range(n):
            res = (res + dp[final_mask][i]) % MOD
        
        return res