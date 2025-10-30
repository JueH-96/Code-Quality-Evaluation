from typing import List

class Solution:
  def minCost(self, nums: List[int], x: int) -> int:
    n = len(nums)
    
    # min_cost_per_type[j] stores the minimum cost found so far
    # to acquire a chocolate of type j. This considers that type j
    # could have been acquired after 0 shifts, or 1 shift, ..., up to the current 's' shifts.
    # Initialize with costs if acquired after 0 shifts (i.e., no operations).
    # Chocolates at index j are initially of type j, costing nums[j].
    min_cost_per_type = list(nums) # Creates a mutable copy of nums.
    
    # current_sum_choc_costs is sum(min_cost_per_type[j] for j in 0..n-1).
    # Initially, this is sum(nums), as min_cost_per_type is a copy of nums.
    current_sum_choc_costs = sum(min_cost_per_type)
    
    # min_overall_cost stores the minimum total cost found across all considered numbers of shifts.
    # Initialize with the total cost for 0 shifts:
    # Cost = (0 operations * cost_per_operation x) + current_sum_choc_costs
    # Since 0 * x = 0, this is just current_sum_choc_costs.
    min_overall_cost = current_sum_choc_costs 
    
    # Iterate through the number of shift operations 's'.
    # 's' represents the total number of shift operations performed if we choose this path.
    # We consider s from 1 up to n-1.
    # s=0 is the base case handled by initialization.
    # s=n is equivalent to s=0 in terms of which chocolate has which type,
    # but s=n incurs n*x operation cost vs 0*x for s=0.
    # So, s >= n doesn't need to be checked if x > 0. If x=0, it's same cost.
    # Thus, iterating up to n-1 is sufficient.
    for s in range(1, n):
      cost_of_shifts = s * x # Cost incurred for performing 's' shift operations.
      
      # For this number of shifts 's', we determine the costs of chocolates
      # if each type 'j' were to be collected after *exactly* 's' shifts.
      # Then, we update min_cost_per_type[j] if this new way is cheaper
      # than any way found for fewer shifts (0 to s-1).
      for j in range(n): # 'j' iterates over chocolate types 0 to n-1.
        # If 's' shifts are performed, the chocolate originally at physical index 'p_orig'
        # (with cost nums[p_orig]) becomes type (p_orig + s) % n.
        # To collect desired type 'j', we need to find the original physical index 'p_orig'
        # such that its chocolate becomes type 'j' after 's' shifts.
        # So, (p_orig + s) % n = j.
        # This implies p_orig = (j - s) % n (using Python's % behavior for negative results,
        # e.g. (0 - 1) % 3 == -1 % 3 == 2).
        # The cost of this specific chocolate item is nums[p_orig].
        cost_type_j_if_obtained_after_s_shifts = nums[(j - s) % n]
        
        if cost_type_j_if_obtained_after_s_shifts < min_cost_per_type[j]:
          # This way of getting type 'j' (by waiting for 's' shifts and picking the
          # chocolate that has become type 'j') is cheaper than any previously known way
          # (i.e., by waiting s' < s shifts).
          
          # Update the sum of chocolate costs: subtract old min_cost_per_type[j], add new one.
          current_sum_choc_costs -= min_cost_per_type[j]
          current_sum_choc_costs += cost_type_j_if_obtained_after_s_shifts
          
          # Update the known minimum cost for type 'j'.
          min_cost_per_type[j] = cost_type_j_if_obtained_after_s_shifts
          
      # After updating all min_cost_per_type for the current 's' (meaning,
      # min_cost_per_type[j] now reflects min cost for type j using up to 's' shifts),
      # calculate the total cost for this 's' scenario.
      # Total cost = (cost of 's' shifts) + (sum of cheapest ways to get each type, given up to 's' shifts are made)
      current_total_cost_for_s_shifts = cost_of_shifts + current_sum_choc_costs
      
      # Update the overall minimum cost if this 's' scenario is better.
      if current_total_cost_for_s_shifts < min_overall_cost:
        min_overall_cost = current_total_cost_for_s_shifts
        
    return min_overall_cost