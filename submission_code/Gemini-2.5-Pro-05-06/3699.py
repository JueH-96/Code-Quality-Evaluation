import collections
from typing import List

class Solution:
  def numberOfSubsequences(self, nums: List[int]) -> int:
    n = len(nums)
    ans = 0
    
    # According to constraints, 1 <= nums[i] <= 1000.
    # We can use a fixed max_val.
    max_numeric_val = 1000 
    # If values could be larger, we would find max_val dynamically:
    # max_val_in_nums = 0
    # if n > 0:
    #   for x in nums:
    #     if x > max_val_in_nums:
    #       max_val_in_nums = x
    # if max_val_in_nums == 0 and n > 0 : # handles if nums might contain 0, problem says positive
    #    pass # max_numeric_val remains 1000 or some default for empty/all-zero array
    # else:
    #    max_numeric_val = max_val_in_nums


    # count_prefix[k][val] = count of val in nums[0...k-1] (i.e., prefix of length k)
    # Size: (n+1) x (max_numeric_val+1)
    count_prefix = [[0] * (max_numeric_val + 1) for _ in range(n + 1)]
    # count_prefix[0] is all zeros (empty prefix)
    for k in range(1, n + 1):
      # Copy counts from prefix of length k-1
      for val_idx in range(1, max_numeric_val + 1):
        count_prefix[k][val_idx] = count_prefix[k-1][val_idx]
      # Add current element nums[k-1]
      if 1 <= nums[k-1] <= max_numeric_val : # defensive check, though problem implies values are in range
          count_prefix[k][nums[k-1]] += 1
      
    # count_suffix[k][val] = count of val in nums[k...n-1] (i.e., suffix starting at index k)
    # Size: (n+1) x (max_numeric_val+1)
    # count_suffix[n] is all zeros (empty suffix)
    count_suffix = [[0] * (max_numeric_val + 1) for _ in range(n + 1)]
    for k in range(n - 1, -1, -1):
      # Copy counts from suffix starting at k+1
      for val_idx in range(1, max_numeric_val + 1):
        count_suffix[k][val_idx] = count_suffix[k+1][val_idx]
      # Add current element nums[k]
      if 1 <= nums[k] <= max_numeric_val: # defensive check
          count_suffix[k][nums[k]] += 1

    # Indices p, q, r, s must satisfy:
    # p < q-1 < q < r-1 < r < s-1 < s
    # Which implies:
    # p_idx <= q_idx - 2
    # q_idx <= r_idx - 2  (or r_idx >= q_idx + 2)
    # r_idx <= s_idx - 2  (or s_idx >= r_idx + 2)
    
    # Loop ranges:
    # q_idx: from 2 (p_idx=0) up to n-5 (s_idx=n-1, r_idx=n-3, q_idx=n-5)
    # r_idx: from q_idx+2 up to n-3 (s_idx=n-1, r_idx=n-3)
    
    # q_idx from 2 up to (n-1)-4 = n-5
    for q_idx in range(2, n - 4): # Python range is exclusive for end, so n-4 means up to n-5
      val_q = nums[q_idx]
      
      # p_idx ranges from 0 to q_idx-2.
      # counts_p stores counts for nums[0...q_idx-2].
      # This corresponds to count_prefix[q_idx-1] (prefix of length q_idx-1).
      # E.g., if q_idx=2, p_idx can be 0. This is nums[0]. Prefix length is 1 (q_idx-1).
      current_counts_p = count_prefix[q_idx-1]
      
      # r_idx from q_idx+2 up to (n-1)-2 = n-3
      for r_idx in range(q_idx + 2, n - 2): # Python range up to n-3
        val_r = nums[r_idx]
        
        # s_idx ranges from r_idx+2 to n-1.
        # counts_s stores counts for nums[r_idx+2...n-1].
        # This corresponds to count_suffix[r_idx+2].
        # If r_idx+2 >= n, this refers to an empty suffix, for which counts are 0.
        # count_suffix[n] is correctly all zeros.
        # max r_idx is n-3. r_idx+2 is n-1. count_suffix[n-1] is valid.
        # if r_idx+2 could be > n, use count_suffix[min(r_idx+2, n)] or handle.
        # Given loop for r_idx, r_idx+2 is at most (n-3)+2 = n-1. So current_counts_s is always valid.
        current_counts_s = count_suffix[r_idx+2]
        
        sum_qr = 0
        # Iterate over possible values of nums[p_idx] (denoted val_p)
        for val_p in range(1, max_numeric_val + 1):
          num_val_p = current_counts_p[val_p]
          if num_val_p == 0: # If this val_p did not appear for p_idx, skip
            continue
          
          # Condition: nums[p] * nums[r] == nums[q] * nums[s]
          # Rewritten: val_p * val_r == val_q * target_val_s
          # So, target_val_s = (val_p * val_r) / val_q
          
          product_pr = val_p * val_r # Max 1000*1000 = 10^6
          if product_pr % val_q == 0: # Check divisibility
            target_val_s = product_pr // val_q
            
            # target_val_s must be in the valid range for element values
            if 1 <= target_val_s <= max_numeric_val:
              num_target_val_s = current_counts_s[target_val_s]
              sum_qr += num_val_p * num_target_val_s
        ans += sum_qr
        
    return ans