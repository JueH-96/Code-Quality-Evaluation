from typing import List

class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        # We only need to go up to 100000 + 1 for safe indexing (e+1 can be 100001)
        MAX_VAL = 100001
        
        # dp[x] = number of good subsequences ending with value x
        # sumdp[x] = sum of all elements in those subsequences ending with x
        dp = [0] * (MAX_VAL + 1)
        sumdp = [0] * (MAX_VAL + 1)
        
        for e in nums:
            # Get dp/sumdp for neighboring values if in range
            left_dp = dp[e - 1] if e > 0 else 0
            left_sum = sumdp[e - 1] if e > 0 else 0
            right_dp = dp[e + 1] if e < MAX_VAL else 0
            right_sum = sumdp[e + 1] if e < MAX_VAL else 0
            
            old_dp_e = dp[e]
            old_sum_e = sumdp[e]
            
            # Update dp[e] (count of subsequences ending with e)
            new_dp_e = (old_dp_e + left_dp + right_dp + 1) % MOD
            
            # Update sumdp[e] (sum of all elements in subsequences ending with e)
            #   - Keep old subsequences ending with e
            #   - Extend subsequences that ended with e-1 or e+1 by adding e
            #   - Start a new single-element subsequence [e]
            new_sum_e = (
                old_sum_e
                + (left_sum + left_dp * e)
                + (right_sum + right_dp * e)
                + e
            ) % MOD
            
            dp[e] = new_dp_e
            sumdp[e] = new_sum_e
        
        # The answer is the sum of sums of all good subsequences (sum over all endpoints)
        return sum(sumdp) % MOD