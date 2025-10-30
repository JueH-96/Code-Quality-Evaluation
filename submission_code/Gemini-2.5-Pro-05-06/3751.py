from typing import List

class Solution:
  def maxFrequency(self, nums: List[int], k: int) -> int:
    n = len(nums)

    initial_k_count = 0
    for val_in_nums in nums:
      if val_in_nums == k:
        initial_k_count += 1
    # Alternative: initial_k_count = nums.count(k)

    # max_overall_freq is initialized with initial_k_count.
    # This covers the scenario where x=0 (i.e., target_val_to_become_k == k).
    # In that case, delta_array would be all zeros, Kadane sum is 0.
    # So, initial_k_count + 0 = initial_k_count is a possible outcome.
    max_overall_freq = initial_k_count

    # Iterate over all possible values (1 to 50) that elements in nums
    # could originally have, to become k after adding x.
    # Let this original value be target_val_to_become_k.
    # Then x = k - target_val_to_become_k.
    for target_val_to_become_k in range(1, 51): # Values are 1 to 50 inclusive.
      
      if target_val_to_become_k == k:
        # This case (x=0) is already handled by initializing max_overall_freq.
        continue

      # Construct delta_array for the current target_val_to_become_k.
      # delta_array[i] represents the change in k-count contribution from nums[i]
      # if nums[i] is part of the chosen subarray.
      delta_array = [0] * n
      for i in range(n):
        val = nums[i]
        if val == target_val_to_become_k:
          # Element `val` is targeted to become `k`.
          # Since target_val_to_become_k != k (due to the `continue` above),
          # `val` was not `k` originally.
          # It changes from non-k to k. Delta contribution: +1.
          delta_array[i] = 1
        elif val == k:
          # Element `val` was `k`.
          # It's not `target_val_to_become_k` (else caught by the first `if`).
          # It will change to `val + x = k + (k - target_val_to_become_k)`.
          # This new value is `k` iff `k - target_val_to_become_k == 0`, i.e., `target_val_to_become_k == k`.
          # But that specific case was skipped by `continue`.
          # So, this element `val` (which is `k`) changes to a non-`k` value. Delta contribution: -1.
          delta_array[i] = -1
        else:
          # Element `val` is neither `target_val_to_become_k` nor `k`.
          # It was not `k`, and it will not become `k`. Delta contribution: 0.
          delta_array[i] = 0
      
      # Kadane's algorithm for non-empty subarray sum.
      # n >= 1, so delta_array is non-empty and delta_array[0] is safe.
      if n == 0: # Should not be reached due to constraints 1 <= n.
          current_kadane_max = 0
      else:
          current_max_ending_here = delta_array[0]
          max_so_far_kadane = delta_array[0]
          for i in range(1, n):
            num_delta = delta_array[i]
            # Current subarray can either start new with num_delta or extend previous.
            current_max_ending_here = max(num_delta, current_max_ending_here + num_delta)
            # Update max_so_far if current_max_ending_here is greater.
            max_so_far_kadane = max(max_so_far_kadane, current_max_ending_here)
          current_kadane_max = max_so_far_kadane
      
      # The sum current_kadane_max can be negative. This is valid.
      # The total frequency for this choice of target_val (and implied x)
      # and optimal subarray is initial_k_count + current_kadane_max.
      max_overall_freq = max(max_overall_freq, initial_k_count + current_kadane_max)
            
    return max_overall_freq