import math

class Solution:
  def _gcd(self, a: int, b: int) -> int:
    return math.gcd(a, b)

  def _lcm(self, a: int, b: int) -> int:
    if a == 0 or b == 0: # Should not be reached if target elements are >= 1
      return 0
    # Optimization for LCM with 1, though standard formula also works.
    if a == 1: return b
    if b == 1: return a
    # abs() isn't necessary as target elements are positive.
    res = (a * b) // self._gcd(a, b)
    return res

  def minimumIncrements(self, nums: list[int], target: list[int]) -> int:
    M = len(target)
    
    # Precompute LCM values for all submasks of target elements
    # L_values[s_mask] stores the LCM of target elements indicated by s_mask
    L_values = [0] * (1 << M) # L_values[0] will remain 0, unused.
    
    for s_mask in range(1, 1 << M):
      elements_in_s_mask = []
      for j in range(M): # j is the index in the target array
        if (s_mask >> j) & 1: # If j-th bit is set in s_mask
          elements_in_s_mask.append(target[j])
      
      # Calculate LCM of elements in elements_in_s_mask
      # LCM of a single number t_i is t_i.
      # LCM of [t1, t2, ..., tk] can be found by iteratively applying lcm(a,b)
      current_val_lcm = elements_in_s_mask[0]
      for k in range(1, len(elements_in_s_mask)):
          current_val_lcm = self._lcm(current_val_lcm, elements_in_s_mask[k])
      L_values[s_mask] = current_val_lcm

    # dp[mask] = minimum cost to cover targets represented by 'mask'
    # dp_curr means "dp state using nums up to the PREVIOUS num_val considered"
    dp_curr = [float('inf')] * (1 << M)
    dp_curr[0] = 0 # Cost to cover empty set of targets is 0

    for num_val in nums:
      # dp_next means "dp state after considering current num_val"
      # Initialize dp_next with dp_curr (handles option to not use current num_val)
      dp_next = list(dp_curr) 

      # Option: use current num_val to cover some non-empty s_mask of targets
      for s_mask in range(1, 1 << M): # Iterate over all non-empty subsets of targets
        L_s = L_values[s_mask]
        
        # Cost to make num_val a multiple of L_s
        cost_for_num_val_to_cover_s_mask = (L_s - (num_val % L_s)) % L_s
        
        # Iterate over all possible masks that could have been covered by previous nums
        for target_mask_prev_covered in range(1 << M):
          if dp_curr[target_mask_prev_covered] == float('inf'):
            # This previous state is unreachable, cannot extend from it
            continue
          
          # If previous nums covered target_mask_prev_covered (cost dp_curr[target_mask_prev_covered]),
          # and current num_val covers s_mask (cost cost_for_num_val_to_cover_s_mask),
          # then the combined mask (target_mask_prev_covered | s_mask) is now covered.
          combined_mask = target_mask_prev_covered | s_mask
          new_cost = dp_curr[target_mask_prev_covered] + cost_for_num_val_to_cover_s_mask
          
          # Update if this path is cheaper
          if new_cost < dp_next[combined_mask]:
            dp_next[combined_mask] = new_cost
            
      # Move to next num_val: current dp_next becomes dp_curr for the next iteration
      dp_curr = dp_next

    # The final answer is the minimum cost to cover all target elements (mask with all M bits set)
    final_ans = dp_curr[(1 << M) - 1]
    
    # According to constraints (target.length <= nums.length), a solution should always exist.
    # Thus, final_ans should not be float('inf').
    return final_ans