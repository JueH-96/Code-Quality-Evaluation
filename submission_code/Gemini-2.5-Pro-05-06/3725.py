import collections
from typing import List

class Solution:
  def _calculate_boundary_indices(self, nums: List[int], is_prev: bool, is_greater: bool, is_strict: bool) -> List[int]:
    n = len(nums)
    # Default boundary: -1 for previous, n for next
    res = [-1 if is_prev else n] * n
    stack = collections.deque()

    # Determine loop range based on is_prev (left-to-right or right-to-left)
    if is_prev:
        loop_range = range(n)
    else:
        loop_range = range(n - 1, -1, -1)

    for i in loop_range:
        # Pop elements from stack based on comparison with nums[i]
        while stack:
            val_stack_top = nums[stack[-1]]
            pop_condition = False
            if is_greater: # Looking for a greater element
                if is_strict: # Strictly greater
                    pop_condition = (val_stack_top <= nums[i])
                else: # Greater or equal
                    pop_condition = (val_stack_top < nums[i])
            else: # Looking for a smaller element
                if is_strict: # Strictly smaller
                    pop_condition = (val_stack_top >= nums[i])
                else: # Smaller or equal
                    pop_condition = (val_stack_top > nums[i])
            
            if pop_condition:
                stack.pop()
            else: # Found boundary or stack top is suitable
                break
        
        if stack: # If stack is not empty, top is the boundary element's index
            res[i] = stack[-1]
        stack.append(i) # Push current index onto stack
            
    return res

  def _count_valid_subarrays_for_pivot(self, max_x: int, max_y: int, k_limit: int) -> int:
    # max_x: number of elements available to the left (0 to max_x)
    # max_y: number of elements available to the right (0 to max_y)
    # k_limit: max length of subarray (x + y + 1 <= k_limit)
    # We sum max(0, min(max_y, k_limit-1-x) + 1) for x from 0 to max_x.

    if k_limit <= 0: # Subarray length must be at least 1 (pivot itself)
        return 0

    total_count = 0
    
    # B = max_y (number of elements to choose on right, excluding pivot)
    # C = k_limit - 1 (max sum of x and y: x + y <= C)
    B = max_y 
    C = k_limit - 1 

    # Since k_limit >= 1, C >= 0.
    # max_x and max_y are counts, so they are >= 0.
    
    # The term min(B, C-x)+1 must be > 0 for a positive contribution.
    # This means min(B, C-x) >= 0. Since B >= 0, we need C-x >= 0, so x <= C.
    # Effective upper limit for x:
    limit_x_effective = min(max_x, C) 
    if limit_x_effective < 0: # No valid x values if this is negative
        return 0

    # Sum consists of two parts, based on behavior of min(B, C-x):
    # Part 1: Summand is B+1. This occurs when C-x >= B  => x <= C-B.
    x_breakpoint = C - B # x value where behavior changes. Can be negative.
    
    # x for Part 1 ranges from 0 to min(limit_x_effective, x_breakpoint).
    upper_x_part1 = min(limit_x_effective, x_breakpoint)
    
    if upper_x_part1 >= 0: # If this range [0, upper_x_part1] is valid
        num_terms_part1 = upper_x_part1 + 1
        total_count += num_terms_part1 * (B + 1)

    # Part 2: Summand is C-x+1. This occurs when C-x < B   => x > C-B.
    # x for Part 2 ranges from max(0, x_breakpoint + 1) to limit_x_effective.
    lower_x_part2 = max(0, x_breakpoint + 1)

    if lower_x_part2 <= limit_x_effective: # If this range is valid
        # Sum (C-x+1) is an arithmetic progression.
        first_term_val = C - lower_x_part2 + 1
        last_term_val  = C - limit_x_effective + 1
        num_terms_part2 = limit_x_effective - lower_x_part2 + 1
        
        total_count += num_terms_part2 * (first_term_val + last_term_val) // 2
        
    return total_count

  def minMaxSubarraySum(self, nums: List[int], k: int) -> int:
    n = len(nums)
    if n == 0: # Constraint: 1 <= nums.length, but good practice.
        return 0
        
    total_sum_val = 0

    # --- Calculate S_max contribution ---
    # For S_max, nums[i] is max (leftmost if tied).
    # Left boundary: Previous Greater or Equal Element (PGEE).
    # Right boundary: Next Greater Element (NGE, strict).
    prev_greater_eq_indices = self._calculate_boundary_indices(nums, is_prev=True, is_greater=True, is_strict=False)
    next_greater_strict_indices = self._calculate_boundary_indices(nums, is_prev=False, is_greater=True, is_strict=True)

    for i in range(n):
        l_bound_idx = prev_greater_eq_indices[i]
        r_bound_idx = next_greater_strict_indices[i]
        
        # max_x: count of elements strictly left of i, within (l_bound_idx, i)
        max_x = i - (l_bound_idx + 1)
        # max_y: count of elements strictly right of i, within (i, r_bound_idx)
        max_y = (r_bound_idx - 1) - i
        
        num_subarrays = self._count_valid_subarrays_for_pivot(max_x, max_y, k)
        total_sum_val += nums[i] * num_subarrays

    # --- Calculate S_min contribution ---
    # For S_min, nums[i] is min (leftmost if tied).
    # Left boundary: Previous Smaller or Equal Element (PSEE).
    # Right boundary: Next Smaller Element (NSE, strict).
    prev_smaller_eq_indices = self._calculate_boundary_indices(nums, is_prev=True, is_greater=False, is_strict=False)
    next_smaller_strict_indices = self._calculate_boundary_indices(nums, is_prev=False, is_greater=False, is_strict=True)

    for i in range(n):
        l_bound_idx = prev_smaller_eq_indices[i]
        r_bound_idx = next_smaller_strict_indices[i]

        max_x = i - (l_bound_idx + 1)
        max_y = (r_bound_idx - 1) - i
        
        num_subarrays = self._count_valid_subarrays_for_pivot(max_x, max_y, k)
        total_sum_val += nums[i] * num_subarrays
        
    return total_sum_val