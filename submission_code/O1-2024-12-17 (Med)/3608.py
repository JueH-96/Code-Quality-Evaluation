class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        import math
        
        MOD = 10**9 + 7
        n = len(nums)
        
        # Precompute gcd for all pairs up to 200 (including gcd with 0)
        MAX_VAL = 200
        gcd_table = [[0]*(MAX_VAL+1) for _ in range(MAX_VAL+1)]
        for x in range(MAX_VAL+1):
            for y in range(MAX_VAL+1):
                if x == 0:
                    gcd_table[x][y] = y
                elif y == 0:
                    gcd_table[x][y] = x
                else:
                    gcd_table[x][y] = math.gcd(x, y)
        
        # We will use a dynamic programming (DP) approach.
        # dp[i][g1][g2][mask] = number of ways to distribute the first i elements
        # into two subsequences S1, S2 (possibly empty so far) such that:
        #   gcd(S1) = g1
        #   gcd(S2) = g2
        #   mask is a bitmask indicating whether S1 is non-empty (bit 0)
        #                                        and S2 is non-empty (bit 1)
        #
        # At the end, we want the sum of dp[n][g][g][3] for g in 1..200,
        # because mask=3 (binary 11) indicates both subsequences are non-empty,
        # and gcd(S1) = gcd(S2) = g.
        
        # For efficiency, we can roll over dp arrays: dp_current -> dp_next
        dp_current = [[[0]*4 for _ in range(MAX_VAL+1)] for _ in range(MAX_VAL+1)]
        dp_current[0][0][0] = 1  # No elements used, gcds = 0,0, both subsequences empty
        
        for x in nums:
            dp_next = [[[0]*4 for _ in range(MAX_VAL+1)] for _ in range(MAX_VAL+1)]
            for g1 in range(MAX_VAL+1):
                for g2 in range(MAX_VAL+1):
                    for mask in range(4):
                        count_ways = dp_current[g1][g2][mask]
                        if count_ways == 0:
                            continue
                        
                        # 1) Do not use the current element
                        dp_next[g1][g2][mask] = (dp_next[g1][g2][mask] + count_ways) % MOD
                        
                        # 2) Put the current element into subsequence S1
                        new_g1 = gcd_table[g1][x]
                        # Set bit 0 in mask (S1 is non-empty) => new_mask
                        new_mask1 = mask | 1
                        dp_next[new_g1][g2][new_mask1] = (dp_next[new_g1][g2][new_mask1] + count_ways) % MOD
                        
                        # 3) Put the current element into subsequence S2
                        new_g2 = gcd_table[g2][x]
                        # Set bit 1 in mask => new_mask
                        new_mask2 = mask | 2
                        dp_next[g1][new_g2][new_mask2] = (dp_next[g1][new_g2][new_mask2] + count_ways) % MOD
            
            dp_current = dp_next
        
        # Sum up dp[n][g][g][3] for g in [1..200]
        ans = 0
        for g in range(1, MAX_VAL+1):
            ans = (ans + dp_current[g][g][3]) % MOD
        
        return ans