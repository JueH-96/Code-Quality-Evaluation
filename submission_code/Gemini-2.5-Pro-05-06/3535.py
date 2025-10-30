from typing import List

class Solution:
  def countOfPairs(self, nums: List[int]) -> int:
    MOD = 10**9 + 7
    # V_MAX is the maximum possible value for any element in nums, as per constraints.
    # arr1[i] can range from 0 up to nums[i], so up to V_MAX.
    V_MAX = 50 

    n = len(nums)

    # dp_prev[val] stores the number of ways to choose arr1[0]...arr1[i-1]
    # such that arr1[i-1] = val, satisfying all conditions up to i-1.
    dp_prev = [0] * (V_MAX + 1)

    # Base case: i = 0
    # arr1[0] can be 'val_arr1_0', where 0 <= val_arr1_0 <= nums[0].
    # arr1[0] also must be <= V_MAX. nums[0] <= V_MAX is guaranteed by constraints.
    for val_arr1_0 in range(nums[0] + 1):
        # val_arr1_0 is inherently <= V_MAX because nums[0] <= V_MAX
        dp_prev[val_arr1_0] = 1
    
    # Iterate for i from 1 to n-1 (representing arr1[i])
    # In each iteration, dp_prev refers to counts for arr1[i-1],
    # and we compute dp_curr for arr1[i].
    for i in range(1, n):
        # dp_curr[val] will store counts for arr1[i] = val
        dp_curr = [0] * (V_MAX + 1)
        
        # Calculate prefix sums for dp_prev (which corresponds to arr1[i-1])
        # ps_prev[k_idx] = sum(dp_prev[l] for l from 0 to k_idx)
        ps_prev = [0] * (V_MAX + 1)
        
        ps_prev[0] = dp_prev[0]
        for k_idx in range(1, V_MAX + 1):
            ps_prev[k_idx] = (ps_prev[k_idx-1] + dp_prev[k_idx]) % MOD
        
        # Iterate over possible values for arr1[i] (current_val_arr1_i)
        # current_val_arr1_i must satisfy 0 <= current_val_arr1_i <= nums[i].
        # The loop range handles this. current_val_arr1_i is also implicitly <= V_MAX.
        for current_val_arr1_i in range(nums[i] + 1):
            # Calculate upper bound for prev_val_arr1_i_minus_1 (k)
            # k <= current_val_arr1_i  (arr1 non-decreasing: arr1[i-1] <= arr1[i])
            # k <= current_val_arr1_i - (nums[i] - nums[i-1]) (derived from arr2 non-increasing)
            limit_for_k = min(current_val_arr1_i, 
                              current_val_arr1_i - nums[i] + nums[i-1])
            
            if limit_for_k < 0:
                # No valid k exists (since k must be >= 0). Sum of dp_prev[0...limit_for_k] is 0.
                dp_curr[current_val_arr1_i] = 0
            else:
                # Sum dp_prev[k_val] for 0 <= k_val <= limit_for_k.
                # This sum is ps_prev[limit_for_k].
                # limit_for_k is guaranteed to be <= V_MAX by its calculation.
                dp_curr[current_val_arr1_i] = ps_prev[limit_for_k]
        
        # Current row computation is done, dp_curr becomes dp_prev for the next iteration.
        dp_prev = dp_curr
            
    # Final result is sum of all dp_prev[val].
    # dp_prev now holds counts for arr1[n-1] = val.
    # We need to sum these up for all possible val (0 to V_MAX).
    total_count = 0
    # Sum entries from dp_prev[0] to dp_prev[V_MAX]
    for count_for_val in dp_prev: 
        total_count = (total_count + count_for_val) % MOD
            
    return total_count