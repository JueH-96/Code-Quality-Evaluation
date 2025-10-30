import math

class Solution:
  def minCostToEqualizeArray(self, nums: list[int], cost1: int, cost2: int) -> int:
    n = len(nums)
    MOD = 10**9 + 7

    if n == 0:
      return 0
    if n == 1:
      # Array is already equal, no cost.
      return 0

    # Calculate min, max, and sum of nums.
    # Constraints: nums[i] >= 1, so max_val_orig will be >= 1.
    max_val_orig = 0
    min_val_orig = float('inf') 
    sum_val_orig = 0
    for x in nums:
      if x > max_val_orig:
        max_val_orig = x
      if x < min_val_orig:
        min_val_orig = x
      sum_val_orig += x
    
    # Case 1: Using operation 1 twice is cheaper or equal to using operation 2 once.
    # (2 * cost1 <= cost2).
    # Effectively, we only use operation 1.
    # Cost = (Total increments needed) * cost1.
    # To minimize cost, minimize total increments. This means target_val = max_val_orig.
    # Total increments = n * max_val_orig - sum_val_orig.
    if 2 * cost1 <= cost2:
      ans = (n * max_val_orig - sum_val_orig) * cost1
      return ans % MOD

    # Case 2: Using operation 2 is strictly cheaper than using operation 1 twice.
    # (cost2 < 2 * cost1)
    # The strategy is to maximize use of operation 2.
    
    # Helper function to calculate cost for a given target_t.
    # Assumes cost2 < 2 * cost1, so op2 is preferred.
    def calculate_cost_for_target(target_t: int) -> int:
      # target_t must be at least max_val_orig.
      if target_t < max_val_orig:
        # This should not be reached if candidate_targets are chosen correctly.
        return float('inf')

      total_increments_needed = n * target_t - sum_val_orig
      # max_increments_for_one_element is for the element that was originally min_val_orig
      max_increments_for_one_element = target_t - min_val_orig
      
      current_cost = 0
      # Subcase 2a: The element needing max_increments_for_one_element is the "bottleneck".
      # Condition: 2 * max_increments_for_one_element > total_increments_needed
      if 2 * max_increments_for_one_element > total_increments_needed:
        op2_count = total_increments_needed - max_increments_for_one_element
        op1_count = 2 * max_increments_for_one_element - total_increments_needed
        current_cost = op2_count * cost2 + op1_count * cost1
      # Subcase 2b: No single element is a bottleneck.
      # Condition: 2 * max_increments_for_one_element <= total_increments_needed
      else:
        op2_count = total_increments_needed // 2
        op1_count = total_increments_needed % 2
        current_cost = op2_count * cost2 + op1_count * cost1
      return current_cost

    min_total_cost = float('inf')
    
    # Candidate target values for T:
    # Always check T = max_val_orig and T = max_val_orig + 1
    candidate_targets = [max_val_orig, max_val_orig + 1]

    # If n > 2, also check around T_switch.
    # T_switch is where 2 * max_increments_for_one_element approx equals total_increments_needed.
    # 2 * (T - min_val_orig) = n * T - sum_val_orig  => T = (sum_val_orig - 2*min_val_orig) / (n-2)
    if n > 2:
      t_switch_numerator = sum_val_orig - 2 * min_val_orig
      t_switch_denominator = n - 2
      
      # (sum_val_orig - 2*min_val_orig) >= (n-2)*min_val_orig. Since min_val_orig >= 1, numerator >= 0.
      # Denominator is positive (n > 2 implies n-2 > 0).
      # Calculate ceil(t_switch_numerator / t_switch_denominator)
      t_switch_candidate = (t_switch_numerator + t_switch_denominator - 1) // t_switch_denominator
      
      # The actual T value for switching must be at least max_val_orig.
      effective_t_switch = max(max_val_orig, t_switch_candidate)
      
      candidate_targets.append(effective_t_switch)
      candidate_targets.append(effective_t_switch + 1)
    
    # Using set to remove duplicates, then iterating
    for t_val in set(candidate_targets):
        min_total_cost = min(min_total_cost, calculate_cost_for_target(t_val))
      
    return min_total_cost % MOD