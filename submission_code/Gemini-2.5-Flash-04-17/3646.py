from typing import List

class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        # The maximum possible value in nums is 10^5.
        # We need to handle values up to 10^5 and their neighbors 10^5 + 1.
        MAX_VAL = 100000 
        
        # dp_count[v] = number of good subsequences ending with value v
        # dp_sum[v] = sum of elements of all good subsequences ending with value v
        # Array size MAX_VAL + 2 to handle values 0 to MAX_VAL+1 for neighbor lookups
        # Index v-1 needs to be checked if v > 0.
        # Index v+1 for v <= MAX_VAL needs to be handle (v+1 <= MAX_VAL + 1).
        # So size MAX_VAL + 2 is correct for indices 0 to MAX_VAL + 1.
        dp_count = [0] * (MAX_VAL + 2)
        dp_sum = [0] * (MAX_VAL + 2)
        
        for current_v in nums:
            # Calculate count and sum for new good subsequences ending specifically with the current_v from nums[i].
            # These new subsequences use nums[i] as the last element.
            
            # Get count and sum from previous good subsequences ending with current_v - 1
            # These use elements from nums[0...i-1]. Their counts/sums are stored in dp_count/dp_sum before processing nums[i].
            prev_count_minus_1 = dp_count[current_v - 1] if current_v > 0 else 0
            prev_sum_minus_1 = dp_sum[current_v - 1] if current_v > 0 else 0
            
            # Get count and sum from previous good subsequences ending with current_v + 1
            # These use elements from nums[0...i-1]. Their counts/sums are stored in dp_count/dp_sum before processing nums[i].
            # current_v + 1 is always <= MAX_VAL + 1 since current_v <= MAX_VAL.
            prev_count_plus_1 = dp_count[current_v + 1]
            prev_sum_plus_1 = dp_sum[current_v + 1]
            
            # Number of new good subsequences ending with the current_v using nums[i].
            # Formed by extending prev_count_minus_1 seqs, prev_count_plus_1 seqs, plus the single seq [current_v].
            current_num_ending_v = (prev_count_minus_1 + prev_count_plus_1 + 1) % MOD
            
            # Sum of elements for these newly formed subsequences ending with current_v using nums[i].
            # Sum from extending v-1: sum_prev_minus_1 + count_prev_minus_1 * current_v
            sum_from_prev_minus_1_ext = (prev_sum_minus_1 + (prev_count_minus_1 * current_v) % MOD) % MOD
            
            # Sum from extending v+1: sum_prev_plus_1 + count_prev_plus_1 * current_v
            sum_from_prev_plus_1_ext = (prev_sum_plus_1 + (prev_count_plus_1 * current_v) % MOD) % MOD
            
            # Sum from the single sequence [current_v]
            sum_from_self = current_v % MOD

            # Total sum of elements from all newly formed good subsequences ending with current_v using nums[i].
            current_sum_ending_v = (sum_from_prev_minus_1_ext + sum_from_prev_plus_1_ext + sum_from_self) % MOD
            
            # Update the total DP state for value current_v.
            # The total count/sum for value current_v using nums[0...i] is the sum of:
            # 1. Existing good subsequences ending with current_v using nums[0...i-1]. (dp_count[current_v] before update)
            # 2. Newly formed good subsequences ending with current_v using nums[i]. (current_num_ending_v)
            
            # Add the newly formed count to the total count for value current_v
            dp_count[current_v] = (dp_count[current_v] + current_num_ending_v) % MOD
            
            # Add the newly formed sum to the total sum for value current_v
            dp_sum[current_v] = (dp_sum[current_v] + current_sum_ending_v) % MOD

        # The total sum of all good subsequences is the sum of dp_sum[v] for all possible ending values v.
        # Possible ending values are from 0 up to MAX_VAL (100000).
        total_sum_all_subsequences = 0
        # Iterate from 0 up to MAX_VAL (inclusive)
        for v in range(MAX_VAL + 1):
             total_sum_all_subsequences = (total_sum_all_subsequences + dp_sum[v]) % MOD
        
        return total_sum_all_subsequences