import math

class Solution:
  def minOperations(self, k: int) -> int:
    if k == 1:
      # Initial array nums = [1]. Sum is 1.
      # If k=1, sum is already >= k. So 0 operations.
      return 0
    
    min_ops = float('inf')
    
    # Iterate over possible values 'v_target' that all elements could be made equal to.
    # 'v_target' is the value of the element we create first, then duplicate.
    # (v_target - 1) operations to make the first element have value 'v_target' (starting from 1).
    # Then we need 'num_elements_m' such elements, such that num_elements_m * v_target >= k.
    # So, num_elements_m = ceil(k / v_target). This requires (num_elements_m - 1) duplication operations.
    # Total operations = (v_target - 1) + (num_elements_m - 1).
    
    # 'v_target' ranges from 1 to k.
    # See explanation above for why v_target > k is not optimal.
    for v_target in range(1, k + 1):
      # Number of operations to change the initial '1' to 'v_target'
      ops_to_increase_to_v_target = v_target - 1
      
      # Number of elements 'num_elements_m' needed if each element has value 'v_target'
      # num_elements_m * v_target >= k  => num_elements_m >= k / v_target
      # num_elements_m = math.ceil(k / v_target)
      # Using integer arithmetic for ceiling: (numerator + denominator - 1) // denominator
      # This is safe as k >= 1 and v_target >= 1.
      num_elements_m = (k + v_target - 1) // v_target
      
      # Number of duplication operations to get 'num_elements_m' elements.
      # We start with 1 element ([v_target]), so (num_elements_m - 1) duplications are needed.
      ops_to_duplicate = num_elements_m - 1
      
      current_total_ops = ops_to_increase_to_v_target + ops_to_duplicate
      
      if current_total_ops < min_ops:
        min_ops = current_total_ops
        
    return min_ops