from typing import List

class Solution:
  def sumOfGoodSubsequences(self, nums: List[int]) -> int:
    MOD = 10**9 + 7
    
    # Maximum possible value for nums[i] is 10^5.
    # When considering nums[i] = v, we might access dp state for v+1.
    # So, dp array size needs to be at least 10^5 + 1 index, i.e. size 10^5 + 2.
    MAX_ELEM_VAL = 100000 
    DP_ARRAY_SIZE = MAX_ELEM_VAL + 2 # Max index needed is MAX_ELEM_VAL + 1
    
    # dp_count[x] stores the number of good subsequences ending with value x,
    # considering elements processed so far from the input `nums`.
    dp_count = [0] * DP_ARRAY_SIZE
    # dp_sum[x] stores the sum of elements of all good subsequences ending with value x,
    # considering elements processed so far.
    dp_sum = [0] * DP_ARRAY_SIZE
    
    # total_ans will store the sum of elements of ALL good subsequences.
    total_ans = 0
    
    for v in nums:
        # Calculate count and sum for new good subsequences that are formed using
        # the current element 'v' as their last element.
        
        # Base case: The subsequence is just [v] itself.
        # This forms 1 new subsequence [v]. Its sum of elements is v.
        # Initialize count and sum for subsequences ending with current 'v'.
        count_for_current_v = 1
        sum_for_current_v = v 
        
        # Case 2: Extend good subsequences ending in v-1.
        # These are of the form S' + [v], where S' is a good subsequence ending in v-1.
        # e.g., [..., v-1, v]
        if v > 0: # v-1 is a valid non-negative index
            prev_val = v - 1
            
            # Number of good subsequences ending in prev_val found so far.
            num_subsequences_ending_prev = dp_count[prev_val]
            # Sum of elements for those subsequences.
            sum_elements_subsequences_ending_prev = dp_sum[prev_val]
            
            # Each of these num_subsequences_ending_prev can be extended by appending 'v'.
            # So, they contribute num_subsequences_ending_prev to count_for_current_v.
            count_for_current_v += num_subsequences_ending_prev
            
            # For the sum:
            # The original sum of elements of these subsequences was sum_elements_subsequences_ending_prev.
            # Each of them has 'v' appended. So, an additional (num_subsequences_ending_prev * v) is added to their total sum.
            sum_contribution_from_prev = sum_elements_subsequences_ending_prev + num_subsequences_ending_prev * v
            sum_for_current_v += sum_contribution_from_prev

        # Case 3: Extend good subsequences ending in v+1.
        # These are of the form S'' + [v], where S'' is a good subsequence ending in v+1.
        # e.g., [..., v+1, v]
        # v+1 can be at most MAX_ELEM_VAL + 1, which is a valid index in dp arrays.
        next_val = v + 1
        
        num_subsequences_ending_next = dp_count[next_val]
        sum_elements_subsequences_ending_next = dp_sum[next_val]
        
        count_for_current_v += num_subsequences_ending_next
        
        sum_contribution_from_next = sum_elements_subsequences_ending_next + num_subsequences_ending_next * v
        sum_for_current_v += sum_contribution_from_next
        
        # Apply modulo after summing up all components.
        # Python's integers handle large intermediate values before the modulo operation.
        count_for_current_v %= MOD
        sum_for_current_v %= MOD
        
        # The sum_for_current_v is the sum of elements of all good subsequences
        # that are formed using the current nums[i]=v as their *last* element.
        # These are all valid good subsequences, so their sum is added to total_ans.
        total_ans = (total_ans + sum_for_current_v) % MOD
        
        # Update the DP states for value 'v'.
        # dp_count[v] and dp_sum[v] accumulate totals for *all* occurrences of 'v' processed so far
        # and all ways to form subsequences ending in 'v'.
        dp_count[v] = (dp_count[v] + count_for_current_v) % MOD
        dp_sum[v] = (dp_sum[v] + sum_for_current_v) % MOD
            
    return total_ans