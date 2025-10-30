import collections
from typing import List

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        n = len(nums)
        MOD = 10**9 + 7
        
        # MAX_ARR1_VAL is the maximum possible value for any arr1[i].
        # Since 0 <= arr1[i] <= nums[i] and max(nums[i]) is 50, MAX_ARR1_VAL is 50.
        MAX_ARR1_VAL = 50 
        
        # dp_current[val] stores the number of ways to form arr1[i] such that arr1[i] = val.
        # Initialize for i = 0 (arr1[0]).
        # For arr1[0], any value from 0 to nums[0] is valid.
        dp_current = [0] * (MAX_ARR1_VAL + 1)
        for val in range(nums[0] + 1):
            dp_current[val] = 1
            
        # Iterate from i = 0 to n-2 to compute dp_next for arr1[i+1]
        for i in range(n - 1):
            # Calculate prefix sums for dp_current to optimize sum queries.
            # prefix_sums[k] stores sum(dp_current[j]) for 0 <= j <= k.
            prefix_sums = [0] * (MAX_ARR1_VAL + 1)
            # The first element needs careful handling if dp_current[0] is 0
            prefix_sums[0] = dp_current[0]
            for k in range(1, MAX_ARR1_VAL + 1):
                prefix_sums[k] = (prefix_sums[k-1] + dp_current[k]) % MOD
            
            # dp_next will store counts for arr1[i+1]. Initialize to zeros.
            dp_next = [0] * (MAX_ARR1_VAL + 1)
            
            # Calculate the offset for the monotonic conditions:
            # arr1[i+1] >= arr1[i] + max(0, nums[i+1] - nums[i])
            offset = max(0, nums[i+1] - nums[i])
            
            # Iterate over possible values for arr1[i+1] (let's call it next_val)
            # next_val must be within [0, nums[i+1]]
            for next_val in range(nums[i+1] + 1):
                # Calculate the upper bound for prev_val (arr1[i]).
                # prev_val must satisfy:
                # 1. 0 <= prev_val <= nums[i] (general constraint for arr1[i])
                # 2. prev_val <= next_val - offset (derived from monotonic conditions)
                # So, prev_val <= min(nums[i], next_val - offset)
                upper_bound_for_prev_val = min(nums[i], next_val - offset)
                
                # If upper_bound_for_prev_val is negative, it means no valid prev_val exists for this next_val.
                # In this case, dp_next[next_val] remains 0 (as initialized).
                if upper_bound_for_prev_val >= 0:
                    # The sum of dp_current values from 0 up to upper_bound_for_prev_val
                    # is directly available from prefix_sums.
                    dp_next[next_val] = prefix_sums[upper_bound_for_prev_val]
            
            # Update dp_current to be dp_next for the next iteration (i+1).
            dp_current = dp_next
            
        # After iterating through all indices, the total count is the sum of all ways
        # to form arr1[n-1], which are stored in the final dp_current array.
        total_count = 0
        for val in range(MAX_ARR1_VAL + 1):
            total_count = (total_count + dp_current[val]) % MOD
            
        return total_count