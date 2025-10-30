class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # dp_count[x] = number of good subsequences (so far) ending with value x
        # dp_sum[x]   = sum of elements over all good subsequences (so far) ending with value x
        
        max_val = max(nums) if nums else 0
        dp_count = [0] * (max_val + 2)  # +2 for safe access of dp_count[x+1]
        dp_sum = [0] * (max_val + 2)
        
        for e in nums:
            # Take old values (before this element is added) so we don't immediately
            # use our just-updated dp's in the same iteration.
            old_count_e = dp_count[e]
            old_sum_e = dp_sum[e]
            
            old_count_em1 = dp_count[e - 1] if e - 1 >= 0 else 0
            old_sum_em1 = dp_sum[e - 1] if e - 1 >= 0 else 0
            
            old_count_ep1 = dp_count[e + 1] if e + 1 <= max_val else 0
            old_sum_ep1 = dp_sum[e + 1] if e + 1 <= max_val else 0
            
            # Update dp_count[e] and dp_sum[e]
            new_count_e = (old_count_e + old_count_em1 + old_count_ep1 + 1) % MOD
            # Summation update:
            #   1) old_sum_e (extending nothing)
            #   2) old_sum_em1 + e*old_count_em1 (extend all ending with e-1)
            #   3) old_sum_ep1 + e*old_count_ep1 (extend all ending with e+1)
            #   4) + e for the new single-element subsequence
            new_sum_e = (old_sum_e
                         + (old_sum_em1 + e * old_count_em1) % MOD
                         + (old_sum_ep1 + e * old_count_ep1) % MOD
                         + e) % MOD
            
            dp_count[e] = new_count_e
            dp_sum[e] = new_sum_e
        
        # The answer is the sum of dp_sum[x] for all x
        # because any good subsequence ends with some value x.
        return sum(dp_sum) % MOD