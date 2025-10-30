import sys

class Solution:
  def minCost(self, nums: list[int]) -> int:
    n = len(nums)
    memo = {}
    
    # sys.setrecursionlimit(...) # Default limit (1000) is usually fine for N/2 depth if N=1000.

    def solve(k: int, p_idx: int | None):
      state = (k, p_idx)
      if state in memo:
        return memo[state]

      # Construct the current_elements_values (first up to 3 elements of the conceptual list)
      # and their original indices in the `nums` array.
      current_elements_values = []
      current_elements_original_indices = [] # Stores original indices for elements in current_elements_values

      if p_idx is not None:
        current_elements_values.append(nums[p_idx])
        current_elements_original_indices.append(p_idx)
      
      # Fill with elements from nums[k:] until we have 3 elements or run out of nums.
      temp_k_fill = k 
      while len(current_elements_values) < 3 and temp_k_fill < n:
        current_elements_values.append(nums[temp_k_fill])
        current_elements_original_indices.append(temp_k_fill)
        temp_k_fill += 1
      
      num_effective_elements = len(current_elements_values)

      # Base Cases (Operation 2: fewer than 3 elements remain)
      if num_effective_elements == 0:
        return 0
      
      if num_effective_elements == 1:
        # Cost is the value of the single element
        memo[state] = current_elements_values[0]
        return current_elements_values[0]
      
      if num_effective_elements == 2:
        # Cost is the maximum of the two elements
        cost = max(current_elements_values[0], current_elements_values[1])
        memo[state] = cost
        return cost

      # Operation 1: num_effective_elements is 3.
      # These are e0, e1, e2 for the current step.
      e0_val, e1_val, e2_val = current_elements_values[0], current_elements_values[1], current_elements_values[2]
      e0_orig_idx, e1_orig_idx, e2_orig_idx = current_elements_original_indices[0], current_elements_original_indices[1], current_elements_original_indices[2]
      
      # suffix_starts_at_k_orig is the original index in `nums` from where the rest of the list is drawn,
      # after e0, e1, e2 are conceptually laid out. This is `temp_k_fill` from the earlier loop.
      suffix_starts_at_k_orig = temp_k_fill
      
      res = float('inf')

      # Helper to make the recursive call, normalizing state if possible.
      # next_suffix_k: the k for the nums[k:] part of the next conceptual list.
      # kept_element_orig_idx: original index of the element kept, which forms the p_idx for the next state.
      def make_recursive_call(next_suffix_k: int, kept_element_orig_idx: int):
        # If the kept element is contiguous with the start of the next_suffix_k,
        # it means the new list is effectively nums[kept_element_orig_idx:].
        # So, the new state is (kept_element_orig_idx, None).
        if kept_element_orig_idx == next_suffix_k - 1:
          return solve(kept_element_orig_idx, None)
        else:
          # There's a "gap": kept_element_orig_idx is not immediately before next_suffix_k.
          # The new list is effectively [nums[kept_element_orig_idx]] + nums[next_suffix_k:].
          # So, the new state is (next_suffix_k, kept_element_orig_idx).
          # (Assert kept_element_orig_idx < next_suffix_k - 1 usually holds here).
          return solve(next_suffix_k, kept_element_orig_idx)

      # Option 1: Remove e0, e1. Keep e2.
      op_cost1 = max(e0_val, e1_val)
      res = min(res, op_cost1 + make_recursive_call(suffix_starts_at_k_orig, e2_orig_idx))

      # Option 2: Remove e0, e2. Keep e1.
      op_cost2 = max(e0_val, e2_val)
      res = min(res, op_cost2 + make_recursive_call(suffix_starts_at_k_orig, e1_orig_idx))
      
      # Option 3: Remove e1, e2. Keep e0.
      op_cost3 = max(e1_val, e2_val)
      res = min(res, op_cost3 + make_recursive_call(suffix_starts_at_k_orig, e0_orig_idx))
      
      memo[state] = res
      return res

    # Initial call: the conceptual list is nums[0:], so k=0 and no p_idx.
    return solve(0, None)