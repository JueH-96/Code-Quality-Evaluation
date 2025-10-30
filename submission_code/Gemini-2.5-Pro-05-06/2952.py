import math

class Solution:
  def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
    n = len(nums1)
    
    # Calculate sum_nums1_initial and sum_nums2_initial
    # These are sum of nums1[i] and nums2[i] at the beginning (t=0)
    sum_val_nums1 = sum(nums1)
    sum_val_nums2 = sum(nums2)
        
    # If the condition is already met at t=0
    if sum_val_nums1 <= x:
      return 0
      
    # Create pairs of (nums2[i], nums1[i])
    # We sort by nums2[i]. Items with higher nums2[i] benefit more from being associated with a larger time slot multiplier.
    # The DP dp[j] will store the maximum reduction if we pick j items.
    # These j items are assigned "time slots" 1, 2, ..., j based on their nums2 values.
    # The item with the k-th smallest nums2 value among the j chosen items gets time slot k.
    # The "value" of item p (with original values N1_p, N2_p) if assigned time slot k is N1_p + N2_p * k.
    items = []
    for i in range(n):
      items.append((nums2[i], nums1[i])) # Store as (nums2_val, nums1_val)
    
    items.sort() # Sorts by nums2_val primarily, then by nums1_val if nums2_val are equal.

    # dp[j] will store the maximum possible "reduction sum" if we choose to reset j items.
    # The reduction sum for j items is sum(N1_chosen[p] + N2_chosen[p] * time_slot_for_p)
    # where time_slot_for_p are 1, 2, ..., j, assigned according to N2_chosen[p] order.
    # dp[0] = 0 (0 reduction for 0 items). dp array size n+1.
    dp = [0] * (n + 1) 

    # Fill dp table
    # Outer loop: iterate through each item P_k = (N2_val, N1_val) from the sorted list
    for k_idx in range(n): # k_idx is the index in the sorted `items` list
      N2_val, N1_val = items[k_idx]
      
      # Inner loop: j_items_chosen is the count of items chosen for reset.
      # Iterate j_items_chosen from n down to 1. This order is important for standard knapsack-like DP.
      # When calculating dp[j_items_chosen] using item P_k (items[k_idx]), 
      # dp[j_items_chosen-1] should be the state *before* considering P_k.
      for j_items_chosen in range(n, 0, -1): 
        # dp[j_items_chosen] can be formed by:
        # 1. Not choosing P_k: dp[j_items_chosen] remains its value from considering previous items up to items[k_idx-1].
        # 2. Choosing P_k: P_k is one of the j_items_chosen items.
        #    It takes the j_items_chosen-th time slot because its N2_val is the largest (or tied for largest) among
        #    the j_items_chosen items if they are chosen from items[0]...items[k_idx] and items[k_idx] is the one
        #    with the highest N2_val among them (due to sorted order and DP construction).
        #    The reduction from P_k is N1_val + N2_val * j_items_chosen.
        #    We add this to dp[j_items_chosen-1] (max reduction from j_items_chosen-1 items chosen from items[0]...items[k_idx-1]).
        dp[j_items_chosen] = max(dp[j_items_chosen], dp[j_items_chosen-1] + N1_val + N2_val * j_items_chosen)
    
    # Iterate through time t_ops (number of operations) from 1 to n
    # For each time t_ops:
    #   The sum of nums1 elements, if no resets were done, would be:
    #     sum_val_nums1 (initial) + t_ops * sum_val_nums2 (sum of increments over t_ops seconds)
    #   The actual sum is this value minus the maximum reduction achievable with t_ops operations.
    #   The dp[t_ops] we calculated represents this maximum reduction (if we perform t_ops operations on t_ops distinct items,
    #   these operations occurring at times 1, 2, ..., t_ops, assigned optimally based on N2 values).
    for t_ops in range(1, n + 1):
      current_sum_if_no_resets = sum_val_nums1 + t_ops * sum_val_nums2
      max_reduction_for_t_ops = dp[t_ops] 
      
      final_sum_at_t_ops = current_sum_if_no_resets - max_reduction_for_t_ops
      
      if final_sum_at_t_ops <= x:
        return t_ops # Found minimum time t_ops
        
    return -1 # No such time t_ops found in [1, n]