class Solution:
  def makeTheIntegerZero(self, num1: int, num2: int) -> int:
    # Iterate k_ops (number of operations) from 1 up to a practical limit.
    # Given constraints and analysis, k_ops up to 61 should be sufficient.
    for k_ops in range(1, 62): 
      target = num1 - k_ops * num2
      
      # If num2 >= 0, target decreases (or stays same if num2=0) as k_ops increases.
      # If target < k_ops, it means k_ops > target.
      # For any k' > k_ops, (num1 - k' * num2) will also be less than k'.
      # So, if target < k_ops for the current k_ops, no solution will be found.
      if num2 >= 0 and target < k_ops:
        return -1 

      # General conditions for a valid k_ops:
      # 1. target >= k_ops (which also implies target >= 0 as k_ops >= 1)
      # 2. popcount(target) <= k_ops
      
      # Check condition 1: target >= k_ops
      # (If num2 < 0, target increases with k_ops. So target < k_ops might be temporary.
      #  In this case, we 'continue' implicitly by not entering the 'if target >= k_ops' block
      #  or by failing inside it, then proceeding to the next k_ops.)
      if target >= k_ops:
        # Calculate popcount (number of set bits)
        # For positive integers, bin(target).count('1') works.
        # target is guaranteed positive here since target >= k_ops and k_ops >= 1.
        num_set_bits = bin(target).count('1')
        
        # Check condition 2: popcount(target) <= k_ops
        if num_set_bits <= k_ops:
          # All conditions met: k_ops <= target and popcount(target) <= k_ops.
          # Since we iterate k_ops upwards, this is the minimum k_ops.
          return k_ops
        
    # If the loop finishes, no solution was found within the k_ops range.
    return -1