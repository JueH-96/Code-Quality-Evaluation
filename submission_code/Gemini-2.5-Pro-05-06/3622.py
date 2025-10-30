import collections

class Solution:
  def maxFrequency(self, nums: list[int], k: int, numOperations: int) -> int:
    # Constraints: 1 <= nums.length <= 10^5
    # 1 <= nums[i] <= 10^5
    # 0 <= k <= 10^5
    # 0 <= numOperations <= nums.length

    min_val_in_nums = nums[0]
    max_val_in_nums = nums[0]
    for x_val in nums:
      if x_val < min_val_in_nums:
        min_val_in_nums = x_val
      if x_val > max_val_in_nums:
        max_val_in_nums = x_val

    # MAX_ELEMENT_VAL is the hard upper limit for values in nums from constraints.
    MAX_ELEMENT_VAL = 10**5 

    # count_arr[v] stores the frequency of value v in the original nums array.
    # Indices 1 to MAX_ELEMENT_VAL correspond to values 1 to MAX_ELEMENT_VAL.
    count_arr = [0] * (MAX_ELEMENT_VAL + 1)
    for x_val in nums:
      count_arr[x_val] += 1

    # prefix_sum_arr[v] stores sum of frequencies of values from 1 up to v.
    # prefix_sum_arr[v] = count_arr[1] + ... + count_arr[v].
    # prefix_sum_arr[0] is 0, for easier range sum calculation.
    prefix_sum_arr = [0] * (MAX_ELEMENT_VAL + 1)
    
    # Element values start from 1 as per constraints nums[i] >= 1.
    if MAX_ELEMENT_VAL >= 1 : # Check to handle if MAX_ELEMENT_VAL could be 0
        prefix_sum_arr[1] = count_arr[1]
        for v_idx in range(2, MAX_ELEMENT_VAL + 1):
            prefix_sum_arr[v_idx] = prefix_sum_arr[v_idx-1] + count_arr[v_idx]

    max_achieved_freq = 0
    if not nums: # Should not happen based on constraints (nums.length >= 1)
        return 0
    
    # Iterate through all relevant candidate target values t_val.
    # Smallest t_val to check: min_val_in_nums - k
    # Largest t_val to check: max_val_in_nums + k
    # Example: nums=[1], k=0. Loop from 1 to 1.
    # Example: nums=[5], k=10. Loop from -5 to 15.
    loop_min_t = min_val_in_nums - k
    loop_max_t = max_val_in_nums + k
    
    for t_val in range(loop_min_t, loop_max_t + 1):
      # Calculate |EQ_T|: number of elements already equal to t_val
      eq_count = 0
      if 1 <= t_val <= MAX_ELEMENT_VAL: # Only values within original num range can have non-zero counts
        eq_count = count_arr[t_val]
      
      # Calculate total elements that can be made t_val:
      # These are elements x from original nums such that t_val - k <= x <= t_val + k.
      # And 1 <= x <= MAX_ELEMENT_VAL since original numbers are in this range.
      # Determine the actual range [L,H] for x from original nums.
      low_bound_for_x = max(1, t_val - k)
      high_bound_for_x = min(MAX_ELEMENT_VAL, t_val + k)
      
      total_can_be_t = 0
      if low_bound_for_x <= high_bound_for_x: # If the range [L,H] is valid
        # Sum count_arr[x] for x in [low_bound_for_x, high_bound_for_x]
        # using prefix_sum_arr. prefix_sum_arr[0] = 0.
        sum_val = prefix_sum_arr[high_bound_for_x]
        if low_bound_for_x > 0: # low_bound_for_x is at least 1 due to max(1, ...)
                                # So low_bound_for_x - 1 is at least 0.
             sum_val -= prefix_sum_arr[low_bound_for_x - 1]
        total_can_be_t = sum_val
        
      # Calculate |CONV_T|: number of elements x != t_val that can be converted to t_val.
      # These are part of total_can_be_t but are not eq_count.
      conv_count = total_can_be_t - eq_count
      
      current_freq = eq_count + min(conv_count, numOperations)
      if current_freq > max_achieved_freq:
        max_achieved_freq = current_freq
        
    # Since nums.length >= 1, max_achieved_freq will be at least 1.
    # When t_val equals one of the numbers in nums, eq_count will be at least 1.
    # For example, if nums=[5], k=0, numOperations=0:
    # t_val=5: eq_count=1. Range for x is [5,5]. total_can_be_t=1. conv_count=0.
    # current_freq = 1 + min(0,0) = 1. max_achieved_freq becomes 1.

    return max_achieved_freq