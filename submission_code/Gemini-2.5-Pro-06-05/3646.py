from collections import defaultdict
from typing import List

class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        
        MOD = 10**9 + 7
        
        # dp_sum[x]: sum of all good subsequences ending with value x
        dp_sum = defaultdict(int)
        # dp_count[x]: count of all good subsequences ending with value x
        dp_count = defaultdict(int)
        
        for num in nums:
            # Contributions from subsequences ending in num-1 and num+1
            count_prev_minus = dp_count[num - 1]
            sum_prev_minus = dp_sum[num - 1]
            
            count_prev_plus = dp_count[num + 1]
            sum_prev_plus = dp_sum[num + 1]
            
            # Calculate total count of new subsequences formed ending with this `num`.
            # This is 1 (for the subsequence `[num]`) plus the counts from extendable subsequences.
            newly_formed_count = (1 + count_prev_minus + count_prev_plus) % MOD
            
            # Calculate the sum of these newly formed subsequences.
            # Sum from extending subsequences ending in num-1:
            # For each of the `count_prev_minus` subsequences, their sum is increased by `num`.
            # Total sum = (sum of old parts) + (sum of new parts)
            #           = sum_prev_minus + (num * count_prev_minus)
            sum_from_minus = (sum_prev_minus + num * count_prev_minus) % MOD
            sum_from_plus = (sum_prev_plus + num * count_prev_plus) % MOD
            
            newly_formed_sum = (num + sum_from_minus + sum_from_plus) % MOD
            
            # Update DP states for `num` by adding the new contributions.
            dp_count[num] = (dp_count[num] + newly_formed_count) % MOD
            dp_sum[num] = (dp_sum[num] + newly_formed_sum) % MOD

        # The total sum is the sum of all values in dp_sum. Each good
        # subsequence is counted exactly once, categorized by its last element.
        total_sum = sum(dp_sum.values()) % MOD
        
        return total_sum