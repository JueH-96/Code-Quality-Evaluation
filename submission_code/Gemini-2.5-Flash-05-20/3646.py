from typing import List

class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # The maximum possible value for nums[i] is 10^5.
        # We need arrays to store DP states up to this value.
        MAX_NUM_VAL = 100000 
        
        # dp_count[val]: Stores the count of all good subsequences ending with 'val'.
        # dp_sum[val]: Stores the sum of all elements in all good subsequences ending with 'val'.
        # Initialize with zeros.
        dp_count = [0] * (MAX_NUM_VAL + 1)
        dp_sum = [0] * (MAX_NUM_VAL + 1)
        
        # total_overall_sum accumulates the sum of elements from all good subsequences found.
        total_overall_sum = 0
        
        # Iterate through each number in the input array.
        for num in nums:
            current_val = num
            
            # --- Calculate contributions for new good subsequences ending with current_val ---
            
            # 1. Start with the subsequence consisting only of current_val itself: [current_val]
            # This is always a good subsequence.
            current_subsequence_count = 1
            current_subsequence_sum = current_val
            
            # 2. Extend from (current_val - 1):
            # If (current_val - 1) is a valid value, we can append current_val
            # to any good subsequence that ends with (current_val - 1).
            if current_val - 1 >= 0:
                prev_count_minus_1 = dp_count[current_val - 1]
                prev_sum_minus_1 = dp_sum[current_val - 1]
                
                # Add count from subsequences extended from (current_val - 1)
                current_subsequence_count = (current_subsequence_count + prev_count_minus_1) % MOD
                
                # Add sum from subsequences extended from (current_val - 1)
                # This sum includes (prev_sum_minus_1) for the elements already in those subsequences,
                # plus (prev_count_minus_1 * current_val) for the newly added current_val to each.
                current_subsequence_sum = (current_subsequence_sum + prev_sum_minus_1 + prev_count_minus_1 * current_val) % MOD
            
            # 3. Extend from (current_val + 1):
            # If (current_val + 1) is a valid value, we can append current_val
            # to any good subsequence that ends with (current_val + 1).
            if current_val + 1 <= MAX_NUM_VAL:
                prev_count_plus_1 = dp_count[current_val + 1]
                prev_sum_plus_1 = dp_sum[current_val + 1]
                
                # Add count from subsequences extended from (current_val + 1)
                current_subsequence_count = (current_subsequence_count + prev_count_plus_1) % MOD
                
                # Add sum from subsequences extended from (current_val + 1)
                current_subsequence_sum = (current_subsequence_sum + prev_sum_plus_1 + prev_count_plus_1 * current_val) % MOD
            
            # --- Update total sum and DP states ---
            
            # The 'current_subsequence_sum' now holds the total sum of elements for all
            # good subsequences that are formed *ending with this specific 'num' from nums[i]*.
            # Add this sum to our overall total.
            total_overall_sum = (total_overall_sum + current_subsequence_sum) % MOD
            
            # Update the DP states for 'current_val'.
            # These new values reflect the combined count/sum of all good subsequences
            # ending with 'current_val' that have been formed up to this point in the array processing.
            dp_count[current_val] = current_subsequence_count
            dp_sum[current_val] = current_subsequence_sum
            
        return total_overall_sum