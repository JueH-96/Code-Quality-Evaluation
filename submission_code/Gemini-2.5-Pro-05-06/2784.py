from typing import List

class Solution:
  def sumOfPower(self, nums: List[int]) -> int:
    MOD = 10**9 + 7
    
    # Sort the numbers. x_0, x_1, ..., x_{n-1}
    # This means x_0 <= x_1 <= ... <= x_{n-1}
    nums.sort()
    n = len(nums)
    
    total_ans = 0
    
    # S_min_prefix (D_j in the explanation)
    # In iteration j (for element nums[j] which is x_j):
    # S_min_prefix stores D_j = sum of minimums of all non-empty subsequences
    # formed from elements {x_0, ..., x_{j-1}}.
    # D_0 = 0.
    # Recurrence: D_{j+1} = (2 * D_j + x_j) % MOD.
    S_min_prefix = 0
    
    # Iterate through each element x_j (nums[j] after sorting)
    # and consider it as the maximum element of a group S.
    for j in range(n):
      x_j = nums[j]
      
      # Contribution from groups where x_j is the maximum element.
      
      # Case 1: The group S is just {x_j}.
      # Power = max({x_j})^2 * min({x_j}) = x_j^2 * x_j = x_j^3.
      term_case1 = pow(x_j, 3, MOD)
      
      # Case 2: The group S is S' U {x_j}, 
      # where S' is a non-empty subset of {x_0, ..., x_{j-1}}.
      # max_strength(S) = x_j.
      # min_strength(S) = min(S').
      # Power = x_j^2 * min(S').
      # Sum of powers for these groups = x_j^2 * (sum of min(S') for all relevant S')
      #                               = x_j^2 * S_min_prefix (which is D_j).
      term_case2_factor = pow(x_j, 2, MOD) # x_j^2
      term_case2 = (term_case2_factor * S_min_prefix) % MOD
      
      # Add contributions for x_j as max to total_ans
      current_sum_for_max_xj = (term_case1 + term_case2) % MOD
      total_ans = (total_ans + current_sum_for_max_xj) % MOD
      
      # Update S_min_prefix for the next iteration (to calculate D_{j+1}).
      # D_{j+1} = (2 * D_j + x_j) % MOD
      S_min_prefix = (2 * S_min_prefix + x_j) % MOD
            
    return total_ans